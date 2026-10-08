"""The PDF Wizard web UI: a FastAPI app for local, single-user use.

Start it through `pdf-wizard serve`, which binds 127.0.0.1 only, prints a one-time
sign-in token, and runs ONE uvicorn process: sessions, jobs, and rate limits live in
this process's memory.

Routes (only GET and POST are allowed anywhere):
    GET  /login                  sign-in page
    POST /login                  sign in with the token
    POST /logout                 end the session and delete its jobs
    GET  /                       upload page and recent jobs
    POST /jobs                   create a fill job (multipart)
    GET  /jobs/{id}              job progress and results
    GET  /jobs/{id}/status       JSON {status, counts}
    GET  /jobs/{id}/download     filled PDF (attachment)
    POST /jobs/{id}/delete       delete a job now
    GET  /static/...             CSS, JS, and images
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import re
import secrets
import time
from collections import Counter
from collections.abc import AsyncIterator, Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jinja2
from fastapi import FastAPI
from fastapi import HTTPException as FastAPIHTTPException
from fastapi.exceptions import RequestValidationError
from starlette.datastructures import FormData, Secret, UploadFile
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, RedirectResponse, Response
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from starlette.types import Scope

from .. import __version__
from ..config import MB, Limits, Settings
from ..models import FieldResult, FieldStatus, FillReport, RejectReason, SourceKind, SourceSummary
from ..privacy import scrub_text
from . import security
from .jobs import Job, JobLimitError, JobPayload, JobRunner, JobStatus, JobStore, UploadItem, sweep_forever
from .security import RateLimiter, SessionRecord, SessionStore, WebPolicy

if TYPE_CHECKING:
    from ..mapping.base import Mapper
    from ..pipeline import FillOutcome

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

SESSION_COOKIE_BASE = "pdf_wizard_session"
SESSION_COOKIE = security.session_cookie_name(SESSION_COOKIE_BASE, https=True)  # "__Host-pdf_wizard_session"
SESSION_COOKIE_HTTP = security.session_cookie_name(SESSION_COOKIE_BASE, https=False)  # plain-HTTP loopback
SID_KEY = "sid"  # the only value the cookie carries once signed in
LOGIN_CSRF_KEY = "login_csrf"  # the only value the cookie carries before sign-in
MIN_LOGIN_TOKEN_CHARS = 32
MAX_SUBMITTED_TOKEN_CHARS = 512

# Browsers send an empty part for every empty <input type="file">, and the upload
# form has three optional file inputs. Empty parts are dropped before the real count.
OPTIONAL_FILE_INPUTS = 3
UPLOAD_MAX_FIELDS = 20
UPLOAD_MAX_PART_BYTES = 64 * 1024
SMALL_FORM_LIMITS: dict[str, int] = {"max_files": 0, "max_fields": 8, "max_part_size": 4 * 1024}
SOURCE_FIELDS = (("data", "data"), ("docs", "doc"), ("references", "reference"))  # (form field, role)
MAX_CSV_ROW = 100_000
UPLOAD_PATH = "/jobs"  # the only route that accepts a large body
SMALL_BODY_MAX_BYTES = 64 * 1024  # every other route: sign-in, sign-out, and delete forms

STATUS_API_PATH = re.compile(r"/jobs/[^/]+/status")
JOB_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{16,64}")
ROW_PATTERN = re.compile(r"[0-9]{1,6}")
SHORT_VALUE_CHARS = 120  # longer values and evidence are cut visually; the full text is in a <details>
SHORT_EVIDENCE_CHARS = 90

DESCRIPTION = "PDF Wizard fills fillable PDF forms from your documents, with evidence for every value."
LOGIN_FAILED = "That token didn't work. Copy it again from the terminal where PDF Wizard is running."
LOGIN_EXPIRED = "This sign-in page had expired. Paste the token again to sign in."
LOGIN_NO_COOKIE = (
    "Your browser didn't send back PDF Wizard's sign-in cookie. Allow cookies for this page, "
    "or use a current Chrome, Edge, or Firefox, then sign in again."
)

JOB_STATUS_LABELS = {
    JobStatus.QUEUED: "Queued",
    JobStatus.RUNNING: "Running",
    JobStatus.DONE: "Done",
    JobStatus.FAILED: "Failed",
}
JOB_STATUS_TEXT = {
    JobStatus.QUEUED: "Waiting for a free slot…",
    JobStatus.RUNNING: "Reading your files and filling the form…",
}
# (status, label, hint) in display order.
FIELD_STATUSES = (
    (FieldStatus.FILLED, "Filled", "Written to the PDF"),
    (FieldStatus.NO_DATA, "No data", "Nothing found in your files"),
    (FieldStatus.REJECTED, "Rejected", "Failed a check, left blank"),
    (FieldStatus.NOT_WRITABLE, "Not writable", "Read-only, signature, or button"),
)
FIELD_STATUS_LABELS = {status.value: label for status, label, _ in FIELD_STATUSES}
REASON_LABELS = {
    RejectReason.INVALID_VALUE: "Invalid value",
    RejectReason.UNVERIFIED: "Evidence not found",
    RejectReason.LOW_CONFIDENCE: "Low confidence",
    RejectReason.UNKNOWN_SOURCE: "Unknown source",
    RejectReason.RENDER_FAILED: "Couldn't be written",
}
SOURCE_KIND_LABELS = {
    SourceKind.SYSTEM: "Facts from PDF Wizard, such as today's date",
    SourceKind.STRUCTURED: "Data file",
    SourceKind.TEXT: "Text",
    SourceKind.EMAIL: "Email",
    SourceKind.IMAGE: "Image",
    SourceKind.PDF: "PDF document",
    SourceKind.REFERENCE_FORM: "Previously filled form",
    SourceKind.NOTE: "Note",
}
UPLOAD_TYPE_MESSAGES = {
    "form": "The form to fill must be a PDF file (.pdf).",
    "data": "Data files must be JSON (.json) or CSV (.csv) text files.",
    "docs": "Documents must be PDF (.pdf), PNG (.png), JPEG (.jpg, .jpeg), text (.txt), or email (.eml) files.",
    "references": "Previously filled forms must be PDF files (.pdf).",
}
BAD_UPLOAD = "Something in that upload isn't right. Check the files and fields, then try again."
ERROR_TEXT = {
    400: ("Something's wrong with that request", "Check the files and fields, then try again."),
    401: ("Please sign in", "Your session has ended. Sign in again with the token from your terminal."),
    403: (
        "That request was blocked",
        "The page may be out of date, or your session changed. Go back, reload the page, and try again.",
    ),
    404: ("Not found", "This page or job doesn't exist, or it has expired."),
    405: ("Not allowed", "This address doesn't accept that kind of request."),
    413: ("Request too large", "That request is larger than this page accepts."),
    429: ("Too many requests", "Please wait a minute, then try again."),
    500: ("Something went wrong", "Please try again. If it keeps happening, restart PDF Wizard."),
    503: ("PDF Wizard is busy", "Please try again in a few minutes."),
}


class UploadRejected(Exception):
    """An upload failed validation. `message` is generic and safe to show."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def create_app(
    settings: Settings,
    *,
    login_token: str,
    mapper_factory: Callable[[], Mapper],
    fill_fn: Callable[..., FillOutcome] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> FastAPI:
    """Builds the web app.

    `login_token` is the one-time sign-in token printed by `pdf-wizard serve`; tokens
    shorter than 32 characters are refused (fail closed). `mapper_factory` makes a
    Mapper for each job, in a worker thread. `fill_fn` defaults to
    `pipeline.fill_form` (imported lazily). `clock` is a monotonic clock in seconds;
    tests inject a fake one.
    """
    if not isinstance(login_token, str) or len(login_token) < MIN_LOGIN_TOKEN_CHARS:
        raise ValueError("login_token must be a random string of at least 32 characters.")
    if login_token != login_token.strip() or len(login_token) > MAX_SUBMITTED_TOKEN_CHARS:
        raise ValueError("login_token must not have surrounding whitespace or be over 512 characters.")

    policy = WebPolicy()
    web = _WebUI(
        settings, login_token=login_token, mapper_factory=mapper_factory, fill_fn=fill_fn, clock=clock, policy=policy
    )
    # No session secret is configured for local mode, so use a random per-process
    # key (never a hardcoded one). Sessions can't outlive or be shared beyond this process.
    log.warning(
        "Sessions use a random per-process key and live in memory: they end when the "
        "server restarts, and only a single server process is supported."
    )
    session_secret = Secret(secrets.token_bytes(32).hex())

    app = FastAPI(
        title="PDF Wizard",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=web.lifespan,
        # Listed outermost first. No CORS middleware: cross-origin access is never allowed.
        middleware=[
            Middleware(security.ErrorBoundaryMiddleware, responder=web.error_for_scope),
            Middleware(security.SecurityHeadersMiddleware),
            # Blocks DNS rebinding: only these Host header values are served.
            Middleware(TrustedHostMiddleware, allowed_hosts=list(security.ALLOWED_HOSTS), www_redirect=False),
            Middleware(security.MethodAllowListMiddleware, responder=web.error_for_scope),
            # Before any rate limit counts the request: other sites can't use the
            # user's browser to spend the user's limits.
            Middleware(security.CrossOriginGuardMiddleware, responder=web.error_for_scope),
            Middleware(
                security.BodySizeLimitMiddleware,
                max_bytes=settings.limits.max_total_bytes + MB,  # room for multipart framing and text fields
                upload_paths=(UPLOAD_PATH,),
                small_max_bytes=SMALL_BODY_MAX_BYTES,
                responder=web.error_for_scope,
            ),
            # Starlette's SessionMiddleware: '__Host-pdf_wizard_session' over HTTPS;
            # 'pdf_wizard_session' over plain-HTTP loopback, where Chrome rejects the
            # '__Host-' prefix. Secure, HttpOnly, SameSite=Strict, Path=/, no Domain.
            Middleware(
                security.SchemeAwareSessionMiddleware,
                secret_key=session_secret,
                cookie_name=SESSION_COOKIE_BASE,
                max_age=int(policy.session_absolute_s),
                same_site="strict",
            ),
            # Inside the session middleware, so requests are counted per session.
            Middleware(security.RateLimitMiddleware, bucket=web.rate_limit_bucket, responder=web.error_for_scope),
        ],
        exception_handlers={
            HTTPException: web.http_error,
            FastAPIHTTPException: web.http_error,
            RequestValidationError: web.validation_error,
            Exception: web.server_error,
        },
        # Keep request data in this process: no OpenTelemetry spans, metrics, or
        # exception logs (they can include messages derived from personal data).
        telemetry={
            "tracing": False,
            "metrics": False,
            "logs": False,
            "operation_spans": False,
            "auto_configure": False,
        },
    )
    app.state.web = web
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    routes: tuple[tuple[str, Callable[..., Any], str], ...] = (
        ("/login", web.login_page, "GET"),
        ("/login", web.login_submit, "POST"),
        ("/logout", web.logout, "POST"),
        ("/", web.index, "GET"),
        (UPLOAD_PATH, web.create_job, "POST"),
        ("/jobs/{job_id}", web.job_page, "GET"),
        ("/jobs/{job_id}/status", web.job_status, "GET"),
        ("/jobs/{job_id}/download", web.download, "GET"),
        ("/jobs/{job_id}/delete", web.delete_job, "POST"),
    )
    for path, endpoint, method in routes:
        app.add_api_route(path, endpoint, methods=[method], include_in_schema=False)
    return app


class _WebUI:
    """State and request handlers of one app instance."""

    def __init__(
        self,
        settings: Settings,
        *,
        login_token: str,
        mapper_factory: Callable[[], Mapper],
        fill_fn: Callable[..., FillOutcome] | None,
        clock: Callable[[], float],
        policy: WebPolicy,
    ) -> None:
        self.settings = settings
        self.policy = policy
        self._login_token = login_token.encode("utf-8")
        self._clock = clock
        self.templates = _build_templates()
        self.jobs = JobStore(
            clock=clock,
            ttl_s=policy.job_ttl_s,
            max_per_owner=policy.max_jobs_per_session,
            max_total=policy.max_jobs_total,
        )
        self.sessions = SessionStore(
            clock=clock,
            idle_s=policy.session_idle_s,
            absolute_s=policy.session_absolute_s,
            max_sessions=policy.max_sessions,
            on_end=self.jobs.delete_owner,
        )
        self.runner = JobRunner(
            self.jobs,
            settings=settings,
            mapper_factory=mapper_factory,
            fill_fn=fill_fn,
            max_concurrent=policy.max_concurrent_jobs,
            timeout_s=settings.request_timeout_s + policy.job_timeout_grace_s,
            clock=clock,
        )
        self.global_limiter = RateLimiter(policy.requests_per_min, 60.0, clock=clock)
        self.status_limiter = RateLimiter(policy.status_polls_per_min, 60.0, clock=clock)
        self.login_limiter = RateLimiter(policy.logins_per_min, 60.0, clock=clock)
        self.upload_limiter = RateLimiter(policy.jobs_per_min, 60.0, clock=clock)
        self._limits_view = _limits_view(settings.limits)

    def rate_limit_bucket(self, scope: Scope) -> tuple[RateLimiter, str] | None:
        """The limiter and key that count a request (runs inside the session middleware).

        Static files aren't counted. The job status API has its own, larger budget
        so open job tabs can poll. Requests are keyed by session ID when the signed
        cookie has one, else by client; requests from other sessions or from
        cookieless clients never use up a signed-in session's budget.
        """
        path = str(scope.get("path", ""))
        if path.startswith(security.STATIC_PREFIX):
            return None  # CSS, JS, and images: no personal data, and cheap to serve
        session = scope.get("session")
        sid = session.get(SID_KEY) if isinstance(session, dict) else None
        key = f"sid:{sid}" if isinstance(sid, str) and sid else f"client:{security.client_key(scope)}"
        limiter = self.status_limiter if STATUS_API_PATH.fullmatch(path) else self.global_limiter
        return limiter, key

    # ----------------------------------------------------------------- lifespan

    @contextlib.asynccontextmanager
    async def lifespan(self, app: FastAPI) -> AsyncIterator[None]:
        self.runner.start()
        sweeper = asyncio.create_task(
            sweep_forever(self.sessions, self.jobs, self.policy.sweep_interval_s), name="pdf-wizard-sweeper"
        )
        try:
            yield
        finally:
            sweeper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sweeper
            await self.runner.shutdown()

    # ------------------------------------------------------------------ errors

    def error_for_scope(self, scope: Scope, status_code: int, headers: Mapping[str, str]) -> Response:
        """Error response for middleware that rejects a request before routing."""
        return self.error_response(Request(scope), status_code, headers)

    def error_response(
        self, request: Request, status_code: int, headers: Mapping[str, str] | None = None
    ) -> Response:
        """The generic error page (JSON for the status API). Never echoes request data."""
        path = request.url.path
        title, message = self._error_text(status_code, path)
        extra = dict(headers or {})
        response: Response
        if STATUS_API_PATH.fullmatch(path):
            response = JSONResponse({"error": message}, status_code=status_code, headers=extra)
        else:
            signed_out = status_code == 401
            response = self._page(
                request,
                "error.html",
                status_code=status_code,
                headers=extra,
                page_title=title,
                title=title,
                message=message,
                code=status_code,
                action_href="/login" if signed_out else "/",
                action_label="Sign in" if signed_out else "Back to PDF Wizard",
            )
        # Also set here: 500 pages from the error boundary (or Starlette's last-resort
        # handler) don't pass through the security headers middleware.
        security.apply_security_headers(response.headers)
        return response

    def _error_text(self, status_code: int, path: str) -> tuple[str, str]:
        if status_code == 413 and path == UPLOAD_PATH:
            limits = self._limits_view
            return (
                "Files too large",
                f"Each file can be up to {limits['file_size']}, and all files together up to "
                f"{limits['total_size']}.",
            )
        if status_code in ERROR_TEXT:
            return ERROR_TEXT[status_code]
        return ERROR_TEXT[500] if status_code >= 500 else ERROR_TEXT[400]

    async def http_error(self, request: Request, exc: Exception) -> Response:
        status_code = int(getattr(exc, "status_code", 500))
        raw_headers = getattr(exc, "headers", None) or {}
        # Pass through only headers this app sets itself; never the exception detail.
        headers = {k: v for k, v in raw_headers.items() if k.lower() in ("allow", "retry-after")}
        return self.error_response(request, status_code, headers)

    async def validation_error(self, request: Request, exc: Exception) -> Response:
        return self.error_response(request, 400)

    async def server_error(self, request: Request, exc: Exception) -> Response:
        log.error("Unhandled error (%s).", type(exc).__name__)
        try:
            return self.error_response(request, 500)
        except Exception:  # noqa: BLE001 - last resort; must not raise
            response = PlainTextResponse("Internal Server Error", status_code=500)
            security.apply_security_headers(response.headers)
            return response

    # ---------------------------------------------------------------- helpers

    def _page(
        self,
        request: Request,
        template: str,
        *,
        page_title: str,
        status_code: int = 200,
        headers: Mapping[str, str] | None = None,
        csrf_token: str | None = None,
        **context: Any,
    ) -> Response:
        values = {
            "page_title": page_title,
            "description": DESCRIPTION,
            "csrf_token": csrf_token,  # set only on signed-in pages; shows the sign-out button
            "version": __version__,
            **context,
        }
        return self.templates.TemplateResponse(request, template, values, status_code=status_code, headers=headers)

    def _session(self, request: Request) -> SessionRecord | None:
        """The signed-in session, or None. Drops a stale or expired session cookie."""
        sid = request.session.get(SID_KEY)
        if not isinstance(sid, str) or not sid:
            return None
        record = self.sessions.get(sid)
        if record is None:
            request.session.clear()
        return record

    @staticmethod
    def _require_csrf(expected: object, form: FormData) -> None:
        if not isinstance(expected, str) or not security.tokens_match(expected, form.get("csrf_token")):
            raise HTTPException(status_code=403)

    @staticmethod
    def _rate_limit(limiter: RateLimiter, key: str) -> None:
        wait = limiter.hit(key)
        if wait is not None:
            raise HTTPException(status_code=429, headers={"Retry-After": security.retry_after(wait)})

    def _owned_job(self, job_id: str, record: SessionRecord) -> Job:
        job = self.jobs.get(job_id, record.sid) if JOB_ID_PATTERN.fullmatch(job_id) else None
        if job is None:
            raise HTTPException(status_code=404)  # same answer for "missing" and "not yours"
        return job

    def _job_view(self, job: Job) -> dict[str, Any]:
        return {
            "id": job.id,
            "form_name": job.form_name or "form.pdf",
            "status": job.status.value,
            "status_label": JOB_STATUS_LABELS[job.status],
            "active": job.active,
            "age": _ago(self._clock() - job.created),
            "expires_in": _remaining(self.jobs.expires_in(job)),
            "error": job.error,
        }

    def _index(
        self, request: Request, record: SessionRecord, *, error: str | None = None, status_code: int = 200
    ) -> Response:
        jobs = [self._job_view(job) for job in self.jobs.list_for(record.sid)]
        return self._page(
            request,
            "index.html",
            status_code=status_code,
            csrf_token=record.csrf_token,
            page_title="Fill a form",
            jobs=jobs,
            max_jobs=self.policy.max_jobs_per_session,
            limits=self._limits_view,
            error=error,
        )

    def _login_form(self, request: Request, *, error: str | None = None, status_code: int = 200) -> Response:
        token = request.session.get(LOGIN_CSRF_KEY)
        if not isinstance(token, str) or not token:
            token = security.new_token()
            request.session[LOGIN_CSRF_KEY] = token
        return self._page(
            request, "login.html", status_code=status_code, page_title="Sign in", login_csrf=token, error=error
        )

    # ----------------------------------------------------------------- routes

    async def login_page(self, request: Request) -> Response:
        if self._session(request) is not None:
            return _see_other("/")
        return self._login_form(request)

    async def login_submit(self, request: Request) -> Response:
        if self._session(request) is not None:
            return _see_other("/")  # already signed in, e.g. from a stale sign-in tab
        async with request.form(**SMALL_FORM_LIMITS) as form:
            if not security.tokens_match(request.session.get(LOGIN_CSRF_KEY), form.get("csrf_token")):
                # Still a 403, but with the sign-in form and a hint, so a stale page
                # or a cookie the browser dropped isn't a dead end.
                return self._login_form(request, error=_login_csrf_error(request), status_code=403)
            submitted = form.get("token")
        # Only failed tokens count, and only after the CSRF check: requests another
        # site can send blind (no cookie, no CSRF token) never use up the attempts.
        # While the budget is spent, no token is checked, not even a correct one.
        limit_key = f"login:{security.client_key(request.scope)}"
        wait = self.login_limiter.retry_in(limit_key)
        if wait is not None:
            raise HTTPException(status_code=429, headers={"Retry-After": security.retry_after(wait)})
        # TODO(security): hosted mode must sign users in through IAP or OAuth 2.0 /
        # OIDC with MFA from the identity provider. MFA, password hashing, and
        # leaked-password checks don't apply to this local, one-time token: it is
        # random (256 bits, never chosen by a person), printed only to the terminal
        # of the user who started the server, and valid only until the server stops.
        valid = (
            isinstance(submitted, str)
            and len(submitted) <= MAX_SUBMITTED_TOKEN_CHARS
            and security.tokens_match(self._login_token, submitted.strip())
        )
        if not valid:
            self.login_limiter.hit(limit_key)
            log.warning("Sign-in failed.")  # never log the submitted value
            return self._login_form(request, error=LOGIN_FAILED, status_code=401)
        # Rotate: a fresh session ID and CSRF token; the old session (if any) ends.
        old_sid = request.session.get(SID_KEY)
        if isinstance(old_sid, str):
            self.sessions.end(old_sid)
        request.session.clear()
        record = self.sessions.create()
        request.session[SID_KEY] = record.sid
        log.info("Sign-in succeeded.")
        return _see_other("/")

    async def logout(self, request: Request) -> Response:
        record = self._session(request)
        if record is None:
            return _see_other("/login")
        async with request.form(**SMALL_FORM_LIMITS) as form:
            self._require_csrf(record.csrf_token, form)
        self.sessions.end(record.sid)  # also deletes the session's jobs
        request.session.clear()
        response = _see_other("/login")  # a full page load also clears client-side state
        response.headers["Clear-Site-Data"] = '"cache", "cookies", "storage"'
        return response

    async def index(self, request: Request) -> Response:
        record = self._session(request)
        if record is None:
            return _see_other("/login")
        return self._index(request, record)

    async def create_job(self, request: Request) -> Response:
        record = self._session(request)
        if record is None:
            return _see_other("/login")
        self._rate_limit(self.upload_limiter, f"jobs:{record.sid}")
        limits = self.settings.limits
        try:
            async with request.form(
                max_files=1 + limits.max_source_files + OPTIONAL_FILE_INPUTS,
                max_fields=UPLOAD_MAX_FIELDS,
                max_part_size=UPLOAD_MAX_PART_BYTES,
            ) as form:
                self._require_csrf(record.csrf_token, form)
                payload = await read_upload(form, limits)
        except UploadRejected as exc:
            return self._index(request, record, error=exc.message, status_code=exc.status_code)
        except HTTPException as exc:
            if exc.status_code != 400:
                raise
            # The multipart parser hit a cap (files, fields, part size) or the body is malformed.
            return self._index(request, record, error=_unreadable_upload(limits), status_code=400)
        try:
            # The name is shown in the UI: mask SSN, card, and email shapes like the report does.
            job = self.jobs.create(record.sid, scrub_text(payload.form_name))
        except JobLimitError as exc:
            return self._index(request, record, error=exc.user_message, status_code=exc.status_code)
        self.runner.submit(job, payload)
        log.info("Job %s queued with %d inputs.", job.id, len(payload.inputs))
        return _see_other(f"/jobs/{job.id}")

    async def job_page(self, request: Request, job_id: str) -> Response:
        record = self._session(request)
        if record is None:
            return _see_other("/login")
        job = self._owned_job(job_id, record)
        report = job.report if job.status is JobStatus.DONE else None
        return self._page(
            request,
            "job.html",
            csrf_token=record.csrf_token,
            page_title=job.form_name or "Job",
            job=self._job_view(job),
            status_text=JOB_STATUS_TEXT.get(job.status, ""),
            status_texts={status.value: text for status, text in JOB_STATUS_TEXT.items()},
            results=_results_view(report) if report is not None else None,
        )

    async def job_status(self, request: Request, job_id: str) -> Response:
        record = self._session(request)
        if record is None:
            raise HTTPException(status_code=401)
        job = self._owned_job(job_id, record)
        counts = _counts(job.report) if job.status is JobStatus.DONE and job.report is not None else {}
        return JSONResponse({"status": job.status.value, "counts": counts})

    async def download(self, request: Request, job_id: str) -> Response:
        record = self._session(request)
        if record is None:
            return _see_other("/login")
        job = self._owned_job(job_id, record)
        if job.status is not JobStatus.DONE or not job.pdf:
            raise HTTPException(status_code=404)
        filename = security.download_name(job.form_name)
        return Response(
            content=job.pdf,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "no-store",
            },
        )

    async def delete_job(self, request: Request, job_id: str) -> Response:
        record = self._session(request)
        if record is None:
            return _see_other("/login")
        async with request.form(**SMALL_FORM_LIMITS) as form:
            self._require_csrf(record.csrf_token, form)
        if not JOB_ID_PATTERN.fullmatch(job_id) or not self.jobs.delete(job_id, record.sid):
            raise HTTPException(status_code=404)
        log.info("Job %s deleted.", job_id)
        return _see_other("/")


# --------------------------------------------------------------------------- #
# Upload parsing
# --------------------------------------------------------------------------- #


async def read_upload(form: FormData, limits: Limits) -> JobPayload:
    """Validates the upload form and reads its files, each with a size cap.

    Raises UploadRejected with a generic message that lists the allowed types.
    Original file names are kept only as display names and parser hints.
    """
    forms = _file_parts(form, "form")
    if not forms:
        raise UploadRejected("Choose the PDF form you want to fill.")
    if len(forms) > 1:
        raise UploadRejected("Choose only one form to fill.")
    sources = [(field, role, upload) for field, role in SOURCE_FIELDS for upload in _file_parts(form, field)]
    if len(sources) > limits.max_source_files:
        raise UploadRejected(f"Add up to {limits.max_source_files} files besides the form.")
    note = _read_note(form, limits)
    row = _read_row(form)
    flatten = _read_flag(form, "flatten")
    if not sources and not note:
        raise UploadRejected("Add at least one data file, document, previously filled form, or note.")

    form_upload = forms[0]
    form_bytes = await _read_capped(form_upload, limits)
    if not security.is_allowed_upload("form", form_upload.filename, form_bytes):
        raise UploadRejected(UPLOAD_TYPE_MESSAGES["form"])
    total = len(form_bytes)
    items: list[UploadItem] = []
    for field, role, upload in sources:
        content = await _read_capped(upload, limits)
        total += len(content)
        if total > limits.max_total_bytes:
            raise UploadRejected(
                f"All files together must be {_human_bytes(limits.max_total_bytes)} or smaller.", 413
            )
        if not security.is_allowed_upload(field, upload.filename, content):
            raise UploadRejected(UPLOAD_TYPE_MESSAGES[field])
        items.append(
            UploadItem(
                role=role,  # type: ignore[arg-type]
                name=security.display_name(upload.filename),
                content=content,
                row=row if role == "data" else None,
            )
        )
    if note:
        items.append(UploadItem(role="note", name="", content=note.encode("utf-8")))
    return JobPayload(
        form=form_bytes,
        form_name=security.display_name(form_upload.filename) or "form.pdf",
        inputs=tuple(items),
        flatten=flatten,
    )


def _file_parts(form: FormData, field: str) -> list[UploadFile]:
    """Uploaded files of a field, without empty file inputs (no name and no bytes)."""
    parts = []
    for value in form.getlist(field):
        if not isinstance(value, UploadFile):
            continue
        if not value.filename and not value.size:
            continue
        parts.append(value)
    return parts


async def _read_capped(upload: UploadFile, limits: Limits) -> bytes:
    data = await upload.read(limits.max_file_bytes + 1)
    if len(data) > limits.max_file_bytes:
        raise UploadRejected(f"Each file must be {_human_bytes(limits.max_file_bytes)} or smaller.", 413)
    return data


def _single_text(form: FormData, name: str) -> str:
    values = form.getlist(name)
    if len(values) > 1 or any(not isinstance(v, str) for v in values):
        raise UploadRejected(BAD_UPLOAD)
    return str(values[0]) if values else ""


def _read_note(form: FormData, limits: Limits) -> str:
    # Browsers submit textarea line breaks as CRLF but count them as one character.
    note = _single_text(form, "notes").replace("\r\n", "\n").replace("\r", "\n").strip()
    if "\x00" in note:
        raise UploadRejected(BAD_UPLOAD)
    if len(note) > limits.max_note_chars:
        raise UploadRejected(f"The note can be up to {limits.max_note_chars:,} characters.")
    return note


def _read_row(form: FormData) -> int | None:
    raw = _single_text(form, "row").strip()
    if not raw:
        return None
    if not ROW_PATTERN.fullmatch(raw) or not 1 <= int(raw) <= MAX_CSV_ROW:
        raise UploadRejected(f"The CSV row must be a whole number from 1 to {MAX_CSV_ROW:,}.")
    return int(raw)


def _read_flag(form: FormData, name: str) -> bool:
    return _single_text(form, name).strip().lower() in ("1", "on", "true", "yes")


def _unreadable_upload(limits: Limits) -> str:
    return (
        f"That upload couldn't be read. Use one form, up to {limits.max_source_files} other files, "
        f"and a note of up to {limits.max_note_chars:,} characters."
    )


# --------------------------------------------------------------------------- #
# View helpers
# --------------------------------------------------------------------------- #


def _build_templates() -> Jinja2Templates:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=True,  # everything shown may come from untrusted files or model output
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return Jinja2Templates(env=env)


def _see_other(url: str) -> RedirectResponse:
    return RedirectResponse(url, status_code=303)


def _login_csrf_error(request: Request) -> str:
    """Hint for a failed sign-in CSRF check. No session cookie at all means the
    browser dropped or blocked it; otherwise the page was stale (e.g. a restart)."""
    has_cookie = SESSION_COOKIE in request.cookies or SESSION_COOKIE_HTTP in request.cookies
    return LOGIN_EXPIRED if has_cookie else LOGIN_NO_COOKIE


def _human_bytes(size: int) -> str:
    if size >= MB:
        value = size / MB
        return f"{value:.0f} MB" if value == int(value) else f"{value:.1f} MB"
    if size >= 1024:
        return f"{size // 1024} KB"
    return f"{size} bytes"


def _limits_view(limits: Limits) -> dict[str, Any]:
    return {
        "max_file_bytes": limits.max_file_bytes,
        "max_total_bytes": limits.max_total_bytes,
        "max_source_files": limits.max_source_files,
        "max_note_chars": limits.max_note_chars,
        "file_size": _human_bytes(limits.max_file_bytes),
        "total_size": _human_bytes(limits.max_total_bytes),
        "note_chars": f"{limits.max_note_chars:,}",
    }


def _ago(seconds: float) -> str:
    if seconds < 60:
        return "just now"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes} min ago"
    return f"{minutes // 60} h ago"


def _remaining(seconds: float) -> str:
    minutes = math.ceil(seconds / 60)
    return "less than a minute" if minutes <= 1 else f"{minutes} min"


def _shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _counts(report: FillReport) -> dict[str, int]:
    if report.counts:
        return {str(status): int(count) for status, count in report.counts.items()}
    return dict(Counter(str(field.status) for field in report.fields))


def _results_view(report: FillReport) -> dict[str, Any]:
    """Display data for a finished job. Values and evidence come from the masked report."""
    counts = _counts(report)
    sources = {source.id: source for source in report.sources}
    total_ms = report.timings_ms.get("total")
    return {
        "stats": [
            {"key": status.value, "label": label, "hint": hint, "count": counts.get(status.value, 0)}
            for status, label, hint in FIELD_STATUSES
        ],
        "rows": [_field_view(result, sources) for result in report.fields],
        "sources": [
            {"id": source.id, "name": source.name, "kind": SOURCE_KIND_LABELS.get(source.kind, "Source")}
            for source in report.sources
        ],
        "warnings": list(report.warnings),
        "meta": [
            ("Model", report.model or "—"),
            ("Prompt version", report.prompt_version or "—"),
            ("Pages", str(report.page_count)),
            ("Flattened", "Yes" if report.flattened else "No"),
            ("Form SHA-256", f"{report.form_sha256[:16]}…" if report.form_sha256 else "—"),
            ("Processing time", f"{total_ms / 1000:.1f} s" if total_ms is not None else "—"),
        ],
    }


def _field_view(result: FieldResult, sources: Mapping[str, SourceSummary]) -> dict[str, Any]:
    confidence = None if result.confidence is None else min(max(float(result.confidence), 0.0), 1.0)
    source = sources.get(result.source_id) if result.source_id else None
    value = result.value or ""
    evidence = result.evidence or ""
    status = str(result.status)
    return {
        "status": status,
        "status_label": FIELD_STATUS_LABELS.get(status, status),
        "label": result.label or result.name,
        "name": result.name,
        "show_name": bool(result.label) and result.label != result.name,
        "page": result.page,
        "value": value,
        "value_short": _shorten(value, SHORT_VALUE_CHARS),
        "value_long": len(value) > SHORT_VALUE_CHARS,
        "sensitive": result.sensitive,
        "source_id": result.source_id,
        "source_name": source.name if source else "",
        "evidence": evidence,
        "evidence_short": _shorten(evidence, SHORT_EVIDENCE_CHARS),
        "evidence_long": len(evidence) > SHORT_EVIDENCE_CHARS,
        "confidence": confidence,
        "confidence_pct": f"{round(confidence * 100)}%" if confidence is not None else "",
        "reason": REASON_LABELS.get(result.reason) if result.reason else None,
        "detail": result.detail,
        "notes": list(result.notes),
    }
