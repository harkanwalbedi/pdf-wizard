"""Security controls for the PDF Wizard web UI (local, single-user mode).

- Response headers (CSP and friends) on every response, including errors and static files.
- HTTP method allow-list, a cross-site request filter (Fetch Metadata and Origin),
  request body caps, and in-memory sliding-window rate limits.
- A server-side session registry: the signed cookie holds only a random session ID
  (plus the pre-login CSRF token), never anything else.
- CSRF tokens (synchronizer pattern) compared in constant time.
- Upload checks: extension allow-list plus content checks (magic bytes / UTF-8 text).

Everything here lives in one process's memory: sessions end when the server
restarts, and limits aren't shared between processes, so the server must run as a
single process bound to 127.0.0.1.

Known limitation: browsers don't isolate cookies by port, so any other service on a
loopback port that the browser visits can read or overwrite the session cookie. Don't
run untrusted local web services while signed in, and sign out when done.

TODO(security): before a hosted (multi-user) mode: replace the one-time local token
with IAP or OAuth 2.0 / OIDC sign-in (verify the IAP JWT with the algorithm pinned to
ES256 and the audience checked), get MFA from the identity provider, move sessions and
rate limits to a shared store, serve over HTTPS with HSTS, and add the upload controls
listed in `is_allowed_upload`.
"""

from __future__ import annotations

import hmac
import ipaddress
import logging
import math
import re
import secrets
from collections import deque
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from starlette.datastructures import Headers, MutableHeaders, Secret
from starlette.exceptions import HTTPException
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import PlainTextResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WebPolicy:
    """Timeouts, caps, and rate limits of the web UI.

    Rate limits count only requests that pass the host, method, and cross-site
    checks. "Per session" falls back to "per client" (all loopback addresses are
    one client) for requests without a session.
    """

    session_idle_s: float = 30 * 60
    session_absolute_s: float = 8 * 60 * 60
    max_sessions: int = 20  # the least recently used session ends when a new one is over the cap
    requests_per_min: int = 300  # per session, every route except /static and the status API
    status_polls_per_min: int = 1200  # per session, the status API only (a job tab polls 40 times a minute)
    logins_per_min: int = 5  # failed sign-ins per client, counted after the CSRF check
    jobs_per_min: int = 6  # per session
    job_ttl_s: float = 60 * 60
    max_jobs_per_session: int = 20
    max_jobs_total: int = 100
    max_concurrent_jobs: int = 2
    job_timeout_grace_s: float = 60  # added to Settings.request_timeout_s
    sweep_interval_s: float = 60


ALLOWED_HOSTS = ("localhost", "127.0.0.1")
ALLOWED_METHODS = frozenset({"GET", "POST"})
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})
# Sec-Fetch-Site values of requests from this origin, or started by the user
# (address bar, bookmark, a link opened from another app).
TRUSTED_FETCH_SITES = frozenset({"same-origin", "none"})
DEFAULT_PORTS: Mapping[str, int] = {"http": 80, "https": 443}
STATIC_PREFIX = "/static/"
_ASCII_DIGITS = re.compile(r"[0-9]+")
_HOST_AND_PORT = re.compile(r"(?P<host>\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9.-]+)(?::(?P<port>[0-9]{1,5}))?")

CONTENT_SECURITY_POLICY = "; ".join(
    (
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self'",
        "img-src 'self'",
        "font-src 'self'",
        "connect-src 'self'",
        "object-src 'none'",
        "frame-ancestors 'none'",
        "base-uri 'none'",
        "form-action 'self'",
    )
)

SECURITY_HEADERS: Mapping[str, str] = {
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    # Not "no-referrer": with it, browsers send `Origin: null` on this app's own form
    # posts (Fetch standard; Blink builds the Origin from the referrer), so the
    # Origin check couldn't tell them from cross-site posts. "same-origin" still
    # sends nothing to other origins.
    "Referrer-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
}

# Renders the generic error page (HTML, or JSON for the status API) for a request
# that is rejected before it reaches a route: (scope, status code, extra headers).
ErrorResponder = Callable[[Scope, int, Mapping[str, str]], Response]


def apply_security_headers(headers: MutableHeaders, *, static: bool = False) -> None:
    """Sets the security headers, replacing any existing values."""
    for name, value in SECURITY_HEADERS.items():
        headers[name] = value
    # Pages and API responses may show personal data: never cache them. Static
    # assets hold no personal data; browsers revalidate them on every use.
    headers["Cache-Control"] = "no-cache" if static else "no-store"


def client_ip(scope: Scope) -> str:
    """The TCP peer address. Never read from X-Forwarded-For: uvicorn runs with
    proxy_headers=False, so the scope holds the real peer."""
    client = scope.get("client")
    return str(client[0]) if client else "unknown"


def client_key(scope: Scope) -> str:
    """Rate-limit key of the TCP peer. All loopback addresses (127.0.0.0/8, ::1)
    are one client: otherwise a local process could use a new source address for
    each request to get fresh limits, or to fill a limiter's key table."""
    peer = client_ip(scope)
    try:
        address = ipaddress.ip_address(peer)
    except ValueError:
        return peer
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return "loopback" if address.is_loopback else str(address)


def session_cookie_name(base_name: str, *, https: bool) -> str:
    """The session cookie's name: '__Host-' prefixed over HTTPS only."""
    return f"__Host-{base_name}" if https else base_name


class SchemeAwareSessionMiddleware:
    """Starlette's SessionMiddleware, with a cookie name that fits the connection.

    Over HTTPS the cookie is '__Host-<name>'. Over plain HTTP (what `pdf-wizard
    serve` uses, on 127.0.0.1 only) it is '<name>': Chromium accepts '__Host-'
    cookies only from https:// URLs, even on loopback, so the prefix would make
    sign-in impossible there. Chrome and Firefox do accept Secure cookies from
    http://127.0.0.1 and http://localhost (potentially trustworthy origins).

    Both variants are Secure, HttpOnly, SameSite=Strict, Path=/, host-only (no
    Domain), signed with the same key, and carry only a random session ID or a
    CSRF token. The scheme comes from the ASGI server (uvicorn runs with
    proxy_headers=False, so request headers can't change it).
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        secret_key: str | Secret,
        cookie_name: str,
        max_age: int,
        same_site: Literal["lax", "strict", "none"] = "strict",
    ) -> None:
        common: dict[str, Any] = {
            "secret_key": secret_key,
            "max_age": max_age,
            "path": "/",
            "same_site": same_site,
            "https_only": True,  # Secure
        }
        self._https = SessionMiddleware(app, session_cookie=session_cookie_name(cookie_name, https=True), **common)
        self._http = SessionMiddleware(app, session_cookie=session_cookie_name(cookie_name, https=False), **common)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        secure = scope.get("scheme") in ("https", "wss")
        await (self._https if secure else self._http)(scope, receive, send)


# --------------------------------------------------------------------------- #
# Pure ASGI middleware
# --------------------------------------------------------------------------- #


class ErrorBoundaryMiddleware:
    """Turns unexpected exceptions into the generic 500 page, logging only the class.

    Install it outermost. Without it, Starlette re-raises the exception to the
    server, which logs the full traceback, and exception messages can contain
    personal data. The responder sets the security headers itself.
    """

    def __init__(self, app: ASGIApp, *, responder: ErrorResponder) -> None:
        self.app = app
        self.responder = responder

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = False

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive, tracking_send)
        except Exception as exc:  # noqa: BLE001 - shown as a generic page; only the class is logged
            status_code = exc.status_code if isinstance(exc, HTTPException) else 500
            if status_code >= 500:
                log.error("Unhandled error (%s).", type(exc).__name__)
            if started:
                return  # part of the response is already sent; the server closes the connection
            try:
                response = self.responder(scope, status_code, {})
            except Exception:  # noqa: BLE001 - last resort; must not raise
                response = PlainTextResponse("Internal Server Error", status_code=500)
                apply_security_headers(response.headers)
            await response(scope, receive, send)


class SecurityHeadersMiddleware:
    """Adds the security headers to every response that passes through it.

    Install it right inside ErrorBoundaryMiddleware so it also covers responses
    from the host, method, cross-site, size, and rate checks. (Error pages set the
    headers themselves too, for responses produced outside this middleware.)
    """

    def __init__(self, app: ASGIApp, *, static_prefix: str = STATIC_PREFIX) -> None:
        self.app = app
        self.static_prefix = static_prefix

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        static = str(scope.get("path", "")).startswith(self.static_prefix)

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                message.setdefault("headers", [])
                apply_security_headers(MutableHeaders(scope=message), static=static)
            await send(message)

        await self.app(scope, receive, send_with_headers)


class MethodAllowListMiddleware:
    """Rejects every HTTP method except GET and POST with 405."""

    def __init__(self, app: ASGIApp, *, responder: ErrorResponder) -> None:
        self.app = app
        self.responder = responder

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope.get("method") not in ALLOWED_METHODS:
            response = self.responder(scope, 405, {"Allow": "GET, POST"})
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


def server_origin(scope: Scope) -> tuple[str, str, int] | None:
    """This server's origin as the browser addressed it: (scheme, host, port) from
    the ASGI scheme and the Host header (already checked by TrustedHostMiddleware)."""
    scheme = str(scope.get("scheme", "")).lower()
    hosts = Headers(scope=scope).getlist("host")
    if scheme not in DEFAULT_PORTS or len(hosts) != 1:
        return None
    return _host_and_port(scheme, hosts[0])


def parse_origin(value: str) -> tuple[str, str, int] | None:
    """(scheme, host, port) of a serialized origin; None for "null" and anything malformed."""
    scheme, separator, rest = value.strip().partition("://")
    scheme = scheme.lower()
    if not separator or scheme not in DEFAULT_PORTS:
        return None
    return _host_and_port(scheme, rest)


def _host_and_port(scheme: str, value: str) -> tuple[str, str, int] | None:
    match = _HOST_AND_PORT.fullmatch(value.strip())
    if match is None:
        return None
    port = int(match["port"]) if match["port"] else DEFAULT_PORTS[scheme]
    if not 0 < port < 65536:
        return None
    return scheme, match["host"].lower(), port


def is_cross_site_request(scope: Scope) -> bool:
    """True for a request that another site (or another loopback port) sent
    through the user's browser, judged by the Fetch Metadata and Origin headers.

    - Unsafe methods (POST): `Sec-Fetch-Site`, when present, must be "same-origin"
      or "none", and `Origin`, when present, must be this server's own origin
      ("null" counts as foreign).
    - Safe methods (GET): a cross-site or same-site request must be a top-level
      navigation (`Sec-Fetch-Mode: navigate`, `Sec-Fetch-Dest: document`), such
      as following a link. Other pages can't load this app's URLs as images,
      frames, or no-cors fetches.

    Requests without these headers (older browsers, scripts) aren't cross-site
    by this test; every POST form still needs its CSRF token.
    """
    headers = Headers(scope=scope)
    sites = headers.getlist("sec-fetch-site")
    origins = headers.getlist("origin")
    if len(sites) > 1 or len(origins) > 1:
        return True
    site = sites[0].strip().lower() if sites else None
    if str(scope.get("method", "")).upper() in SAFE_METHODS:
        if site is None or site in TRUSTED_FETCH_SITES:
            return False
        navigation = (
            headers.get("sec-fetch-mode", "").strip().lower() == "navigate"
            and headers.get("sec-fetch-dest", "").strip().lower() == "document"
        )
        return not navigation
    if site is not None and site not in TRUSTED_FETCH_SITES:
        return True
    if origins:
        own = server_origin(scope)
        return own is None or parse_origin(origins[0]) != own
    return False


class CrossOriginGuardMiddleware:
    """Rejects cross-site requests (see `is_cross_site_request`) with 403.

    Install it inside TrustedHostMiddleware and before every rate limiter, so
    requests that a page on another site makes through the user's browser are
    never counted against the user (blind form posts, no-cors fetches, images).
    """

    def __init__(self, app: ASGIApp, *, responder: ErrorResponder) -> None:
        self.app = app
        self.responder = responder

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and is_cross_site_request(scope):
            log.debug("Blocked a cross-site request.")
            await self.responder(scope, 403, {})(scope, receive, send)
            return
        await self.app(scope, receive, send)


# Picks the limiter and key that count a request, or None for an exempt request.
RateLimitBucket = Callable[[Scope], "tuple[RateLimiter, str] | None"]


class RateLimitMiddleware:
    """Counts every request against the limiter and key that `bucket` picks.

    Install it inside the session middleware so `bucket` can key requests by
    session ID: one browser's requests then never use up another session's limit.
    """

    def __init__(self, app: ASGIApp, *, bucket: RateLimitBucket, responder: ErrorResponder) -> None:
        self.app = app
        self.bucket = bucket
        self.responder = responder

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            choice = self.bucket(scope)
            if choice is not None:
                limiter, key = choice
                wait = limiter.hit(key)
                if wait is not None:
                    response = self.responder(scope, 429, {"Retry-After": retry_after(wait)})
                    await response(scope, receive, send)
                    return
        await self.app(scope, receive, send)


class BodySizeLimitMiddleware:
    """Caps the request body.

    Requests to `upload_paths` may send up to `max_bytes`; all other requests up
    to `small_max_bytes` (the sign-in, sign-out, and delete forms are tiny, and
    urlencoded bodies aren't bounded by the multipart parser's caps). Requests
    that declare a larger Content-Length are rejected before the app runs.
    Streamed bodies (chunked, or a lying Content-Length) are counted as they are
    read; going over raises HTTPException(413) from `receive`, inside the route,
    so Starlette's exception middleware renders the error page.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_bytes: int,
        responder: ErrorResponder,
        upload_paths: Collection[str] = (),
        small_max_bytes: int | None = None,
    ) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.small_max_bytes = max_bytes if small_max_bytes is None else min(small_max_bytes, max_bytes)
        self.upload_paths = frozenset(upload_paths)
        self.responder = responder

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = self.max_bytes if scope.get("path") in self.upload_paths else self.small_max_bytes
        raw_length = Headers(scope=scope).get("content-length")
        if raw_length is not None:
            value = raw_length.strip()
            if not _ASCII_DIGITS.fullmatch(value):
                await self.responder(scope, 400, {})(scope, receive, send)
                return
            if len(value) > 18 or int(value) > limit:
                await self.responder(scope, 413, {})(scope, receive, send)
                return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise HTTPException(status_code=413)
            return message

        await self.app(scope, limited_receive, send)


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #


class RateLimiter:
    """Sliding-window rate limiter kept in memory (not shared across processes).

    Keys are built by the server from trusted values only: the session ID from the
    signed session cookie, or the peer address from the ASGI scope (`client_key`,
    one key for all loopback addresses). The key table is capped; when it is full
    of live keys, new keys are refused (fail closed).
    """

    def __init__(
        self, limit: int, window_s: float, *, clock: Callable[[], float], max_keys: int = 10_000
    ) -> None:
        if limit < 1 or window_s <= 0 or max_keys < 1:
            raise ValueError("limit, window_s, and max_keys must be positive")
        self._limit = limit
        self._window = float(window_s)
        self._clock = clock
        self._max_keys = max_keys
        self._hits: dict[str, deque[float]] = {}

    def hit(self, key: str) -> float | None:
        """Counts one request for `key`. Returns None if allowed, else seconds to wait."""
        now = self._clock()
        hits = self._hits.get(key)
        if hits is None:
            if len(self._hits) >= self._max_keys:
                self._prune(now)
                if len(self._hits) >= self._max_keys:
                    return self._window
            hits = self._hits[key] = deque()
        self._drop_old(hits, now)
        if len(hits) >= self._limit:
            return max(hits[0] + self._window - now, 0.0)
        hits.append(now)
        return None

    def retry_in(self, key: str) -> float | None:
        """Like `hit`, but counts nothing: None if `key` may go ahead now, else
        seconds to wait. For limits on failures, which are counted with `hit`."""
        now = self._clock()
        hits = self._hits.get(key)
        if hits is None:
            if len(self._hits) >= self._max_keys:
                self._prune(now)
                if len(self._hits) >= self._max_keys:
                    return self._window  # no room to count this key: refuse (fail closed)
            return None
        self._drop_old(hits, now)
        if len(hits) >= self._limit:
            return max(hits[0] + self._window - now, 0.0)
        return None

    def _drop_old(self, hits: deque[float], now: float) -> None:
        cutoff = now - self._window
        while hits and hits[0] <= cutoff:
            hits.popleft()

    def _prune(self, now: float) -> None:
        for key in list(self._hits):
            hits = self._hits[key]
            self._drop_old(hits, now)
            if not hits:
                del self._hits[key]


def retry_after(seconds: float) -> str:
    """Value for a Retry-After header: whole seconds, at least 1."""
    return str(max(1, math.ceil(seconds)))


# --------------------------------------------------------------------------- #
# Tokens and sessions
# --------------------------------------------------------------------------- #


def new_token() -> str:
    """A 256-bit random token from the OS CSPRNG (session IDs, CSRF tokens)."""
    return secrets.token_urlsafe(32)


def tokens_match(expected: str | bytes | None, submitted: object) -> bool:
    """Constant-time comparison. False when either side is missing or empty."""
    if not isinstance(expected, (str, bytes)) or not expected:
        return False
    if not isinstance(submitted, str) or not submitted:
        return False
    expected_bytes = expected.encode("utf-8") if isinstance(expected, str) else expected
    return hmac.compare_digest(expected_bytes, submitted.encode("utf-8"))


@dataclass
class SessionRecord:
    sid: str = field(repr=False)
    created: float
    last_seen: float
    csrf_token: str = field(repr=False)
    authenticated: bool = True


class SessionStore:
    """Server-side session registry keyed by random session IDs.

    The signed session cookie (Starlette's SessionMiddleware) carries only the
    session ID, so ending a session here invalidates the cookie immediately.
    `on_end(sid)` runs whenever a session ends (logout, expiry, eviction) so its
    jobs can be deleted with it.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float],
        idle_s: float,
        absolute_s: float,
        max_sessions: int,
        on_end: Callable[[str], object],
    ) -> None:
        self._clock = clock
        self._idle_s = idle_s
        self._absolute_s = absolute_s
        self._max_sessions = max_sessions
        self._on_end = on_end
        self._records: dict[str, SessionRecord] = {}

    def create(self) -> SessionRecord:
        """Starts an authenticated session with a fresh ID and CSRF token."""
        self.purge_expired()
        while len(self._records) >= self._max_sessions:
            oldest = min(self._records.values(), key=lambda r: r.last_seen)
            log.info("Session cap reached; ending the least recently used session.")
            self.end(oldest.sid)
        now = self._clock()
        record = SessionRecord(sid=new_token(), created=now, last_seen=now, csrf_token=new_token())
        self._records[record.sid] = record
        return record

    def get(self, sid: str) -> SessionRecord | None:
        """The live, authenticated session for `sid`, with its idle timer refreshed.

        Expired sessions are ended (with their jobs) and reported as missing.
        """
        record = self._records.get(sid)
        if record is None:
            return None
        now = self._clock()
        if not record.authenticated or self._expired(record, now):
            self.end(sid)
            return None
        record.last_seen = now
        return record

    def end(self, sid: str) -> None:
        if self._records.pop(sid, None) is not None:
            self._on_end(sid)

    def purge_expired(self) -> int:
        now = self._clock()
        expired = [r.sid for r in self._records.values() if self._expired(r, now)]
        for sid in expired:
            self.end(sid)
        return len(expired)

    def __len__(self) -> int:
        return len(self._records)

    def _expired(self, record: SessionRecord, now: float) -> bool:
        return now - record.last_seen > self._idle_s or now - record.created > self._absolute_s


# --------------------------------------------------------------------------- #
# Uploads
# --------------------------------------------------------------------------- #

PDF_MAGIC = b"%PDF-"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"
TEXT_EXTENSIONS = frozenset({".json", ".csv", ".txt", ".eml"})

# Upload form field -> allowed file extensions.
UPLOAD_TYPES: Mapping[str, frozenset[str]] = {
    "form": frozenset({".pdf"}),
    "data": frozenset({".json", ".csv"}),
    "docs": frozenset({".pdf", ".png", ".jpg", ".jpeg", ".txt", ".eml"}),
    "references": frozenset({".pdf"}),
}

_PATH_SEPARATORS = re.compile(r"[\\/]")
# Control characters, zero-width and bidi-override characters (e.g. "evil\u202efdp.exe").
_UNSAFE_DISPLAY_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f\u200b-\u200f\u2028-\u202e\u2060-\u2069\ufeff]")
_UNSAFE_DOWNLOAD_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_SEPARATOR_RUNS = re.compile(r"[_-]*_[_-]*")  # runs of '_'/'-' that include a replaced character


def file_extension(filename: str | None) -> str:
    """Lower-cased extension of the last path component ("" if none)."""
    base = _PATH_SEPARATORS.split(filename or "")[-1]
    dot = base.rfind(".")
    return base[dot:].lower() if dot > 0 else ""


def is_utf8_text(content: bytes) -> bool:
    if b"\x00" in content:
        return False
    try:
        content.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def content_matches(extension: str, content: bytes) -> bool:
    """True if the bytes look like the type the extension claims."""
    if extension == ".pdf":
        return PDF_MAGIC in content[:1024]  # PDF readers accept the header anywhere in the first 1 KB
    if extension == ".png":
        return content.startswith(PNG_MAGIC)
    if extension in (".jpg", ".jpeg"):
        return content.startswith(JPEG_MAGIC)
    if extension in TEXT_EXTENSIONS:
        return is_utf8_text(content)
    return False


def is_allowed_upload(field_name: str, filename: str | None, content: bytes) -> bool:
    """Checks an uploaded file against the allow-list for its form field: the
    extension must be allowed AND the content must match it.

    The parsers in `pdf_wizard.sources` and `pdf_wizard.pdf` validate structure
    further, under size, page, and character caps, and never run PDF JavaScript.

    TODO(security): add malware scanning and CDR (strip JavaScript, embedded files,
    and other active content from PDFs) through an external service before a hosted
    mode; both need infrastructure this local tool doesn't have.
    """
    extension = file_extension(filename)
    allowed = UPLOAD_TYPES.get(field_name, frozenset())
    return bool(content) and extension in allowed and content_matches(extension, content)


def display_name(filename: str | None, *, max_chars: int = 120) -> str:
    """A file name that is safe to show (and autoescape) in the UI.

    Only for display and as a hint to the source parsers: never used in any path.
    """
    base = _PATH_SEPARATORS.split(filename or "")[-1]
    base = " ".join(_UNSAFE_DISPLAY_CHARS.sub("", base).split())
    if len(base) > max_chars:
        extension = file_extension(base)[:10]
        base = base[: max_chars - len(extension) - 1].rstrip() + "…" + extension
    return base


def download_name(form_name: str | None) -> str:
    """'<name>.filled.pdf' built from an allow-list of characters ([A-Za-z0-9._-])."""
    base = _PATH_SEPARATORS.split(form_name or "")[-1]
    if base.lower().endswith(".pdf"):
        base = base[:-4]
    stem = _UNSAFE_DOWNLOAD_CHARS.sub("_", base)
    # Masked names leave runs like "_-_-": "application ***-**-6789" -> "application_6789".
    stem = _SEPARATOR_RUNS.sub("_", stem).strip("._-")[:80].strip("._-")
    return f"{stem}.filled.pdf" if stem else "filled.pdf"
