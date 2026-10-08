"""In-memory job store and background runner for the web UI.

Jobs live only in this process's memory (no job directory, so no files are left
behind) and expire after an hour. While a request is parsed, Starlette spools
uploads over 1 MB to anonymous temporary files, which are deleted when the
request ends. Every lookup checks the owner inside the store, so a route can't
forget the check, and "not found" and "not yours" look the same.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import logging
import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from ..config import Settings
from ..errors import PdfWizardError
from ..models import FillReport
from ..privacy import scrub_text

if TYPE_CHECKING:
    from ..mapping.base import Mapper
    from ..pipeline import FillOutcome
    from .security import SessionStore

log = logging.getLogger(__name__)

GENERIC_FAILURE = "Something went wrong while filling the form."
TIMEOUT_FAILURE = "The job took too long and was stopped."
MAX_ERROR_CHARS = 300


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True)
class UploadItem:
    """One validated input file or note, before conversion to `sources.SourceInput`."""

    role: Literal["data", "doc", "reference", "note"]
    name: str = field(repr=False)  # display name; a hint for the parsers, never a path
    content: bytes = field(repr=False)
    row: int | None = None


@dataclass(frozen=True)
class JobPayload:
    """Everything a job needs to run. Dropped as soon as the job finishes."""

    form: bytes = field(repr=False)
    form_name: str = field(repr=False)
    inputs: tuple[UploadItem, ...] = field(repr=False)
    flatten: bool = False


@dataclass(eq=False)
class Job:
    id: str
    owner: str = field(repr=False)  # session ID of the creator
    created: float
    form_name: str = field(repr=False)  # display name of the uploaded form
    status: JobStatus = JobStatus.QUEUED
    report: FillReport | None = field(default=None, repr=False)  # masked
    pdf: bytes | None = field(default=None, repr=False)
    error: str | None = None  # safe user message
    finished: float | None = None

    @property
    def active(self) -> bool:
        return self.status in (JobStatus.QUEUED, JobStatus.RUNNING)


class JobLimitError(Exception):
    """A job cap was reached. `user_message` is safe to show."""

    def __init__(self, user_message: str, status_code: int) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.status_code = status_code


def _same_owner(owner: str, sid: str) -> bool:
    return hmac.compare_digest(owner.encode("utf-8"), sid.encode("utf-8"))


class JobStore:
    """Jobs by ID, with per-session and total caps and a time to live."""

    def __init__(
        self, *, clock: Callable[[], float], ttl_s: float, max_per_owner: int, max_total: int
    ) -> None:
        self._clock = clock
        self._ttl_s = ttl_s
        self._max_per_owner = max_per_owner
        self._max_total = max_total
        self._jobs: dict[str, Job] = {}

    def create(self, owner: str, form_name: str) -> Job:
        """Adds a queued job. Raises JobLimitError when a cap is reached."""
        self.purge_expired()
        if sum(1 for job in self._jobs.values() if _same_owner(job.owner, owner)) >= self._max_per_owner:
            raise JobLimitError(
                f"You can keep up to {self._max_per_owner} jobs at a time. "
                "Delete a job, or wait for older ones to expire.",
                429,
            )
        if len(self._jobs) >= self._max_total:
            raise JobLimitError("PDF Wizard is busy with other jobs. Try again in a few minutes.", 503)
        job = Job(id=secrets.token_urlsafe(16), owner=owner, created=self._clock(), form_name=form_name)
        self._jobs[job.id] = job
        return job

    def get(self, job_id: str, owner: str) -> Job | None:
        """The job if it exists, hasn't expired, and belongs to `owner`; else None."""
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if self._expired(job, self._clock()):
            del self._jobs[job_id]
            return None
        return job if _same_owner(job.owner, owner) else None

    def delete(self, job_id: str, owner: str) -> bool:
        """Deletes the job if `owner` may see it. Returns False otherwise."""
        if self.get(job_id, owner) is None:
            return False
        del self._jobs[job_id]
        return True

    def list_for(self, owner: str) -> list[Job]:
        """The owner's jobs, newest first."""
        self.purge_expired()
        owned = (job for job in self._jobs.values() if _same_owner(job.owner, owner))
        return sorted(owned, key=lambda job: job.created, reverse=True)

    def delete_owner(self, owner: str) -> int:
        """Deletes every job of a session (logout, expiry)."""
        doomed = [job_id for job_id, job in self._jobs.items() if _same_owner(job.owner, owner)]
        for job_id in doomed:
            del self._jobs[job_id]
        return len(doomed)

    def purge_expired(self) -> int:
        now = self._clock()
        doomed = [job_id for job_id, job in self._jobs.items() if self._expired(job, now)]
        for job_id in doomed:
            del self._jobs[job_id]
        return len(doomed)

    def contains(self, job: Job) -> bool:
        return self._jobs.get(job.id) is job

    def expires_in(self, job: Job) -> float:
        return max(0.0, job.created + self._ttl_s - self._clock())

    def __len__(self) -> int:
        return len(self._jobs)

    def _expired(self, job: Job, now: float) -> bool:
        return now - job.created >= self._ttl_s


class JobRunner:
    """Runs fills in worker threads, at most `max_concurrent` at a time.

    `fill_fn` has the signature of `pipeline.fill_form` and defaults to it. The
    core modules are imported lazily, inside the worker thread: they are heavy, and
    the web package must import even when they are unavailable.
    """

    def __init__(
        self,
        store: JobStore,
        *,
        settings: Settings,
        mapper_factory: Callable[[], Mapper],
        fill_fn: Callable[..., FillOutcome] | None,
        max_concurrent: int,
        timeout_s: float,
        clock: Callable[[], float],
    ) -> None:
        self._store = store
        self._settings = settings
        self._mapper_factory = mapper_factory
        self._fill_fn = fill_fn
        self._max_concurrent = max_concurrent
        self._timeout_s = timeout_s
        self._clock = clock
        self._semaphore: asyncio.Semaphore | None = None
        self._tasks: set[asyncio.Task[None]] = set()  # strong refs, so tasks aren't garbage collected

    def start(self) -> None:
        """Call from the running event loop (app lifespan)."""
        self._semaphore = asyncio.Semaphore(self._max_concurrent)

    def submit(self, job: Job, payload: JobPayload) -> None:
        if self._semaphore is None:
            self.start()
        task = asyncio.get_running_loop().create_task(self._run(job, payload), name=f"pdf-wizard-job-{job.id}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def shutdown(self) -> None:
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _run(self, job: Job, payload: JobPayload) -> None:
        assert self._semaphore is not None
        try:
            async with self._semaphore:
                if not self._store.contains(job):
                    return  # deleted, or its session ended, while queued
                job.status = JobStatus.RUNNING
                outcome = await self._call_with_timeout(payload)
        except PdfWizardError as exc:
            self._fail(job, exc.user_message, type(exc).__name__)
        except TimeoutError:
            self._fail(job, TIMEOUT_FAILURE, "TimeoutError")
        except Exception as exc:  # noqa: BLE001 - shown as a generic message; only the class is logged
            self._fail(job, GENERIC_FAILURE, type(exc).__name__)
        else:
            self._finish(job, outcome)

    async def _call_with_timeout(self, payload: JobPayload) -> FillOutcome:
        worker = asyncio.ensure_future(asyncio.to_thread(self._work, payload))
        try:
            return await asyncio.wait_for(asyncio.shield(worker), timeout=self._timeout_s)
        except TimeoutError:
            # A thread can't be interrupted. Hold this concurrency slot until the
            # worker really ends, so no more than `max_concurrent` fills ever run.
            with contextlib.suppress(Exception):
                await worker
            raise

    def _work(self, payload: JobPayload) -> FillOutcome:
        from ..pipeline import FillOptions
        from ..sources import SourceInput

        fill = self._fill_fn
        if fill is None:
            from ..pipeline import fill_form as fill
        inputs = [SourceInput(role=i.role, name=i.name, content=i.content, row=i.row) for i in payload.inputs]
        return fill(
            payload.form,
            inputs,
            mapper=self._mapper_factory(),
            settings=self._settings,
            options=FillOptions(flatten=payload.flatten),
            form_name=payload.form_name,
        )

    def _finish(self, job: Job, outcome: FillOutcome) -> None:
        report = getattr(outcome, "report", None)
        pdf = getattr(outcome, "pdf", None)
        # Fail closed: only a masked report may ever be displayed.
        if not isinstance(report, FillReport) or not report.masked or not isinstance(pdf, bytes) or not pdf:
            self._fail(job, GENERIC_FAILURE, "UnexpectedOutcome")
            return
        job.report = report
        job.pdf = pdf
        job.status = JobStatus.DONE
        job.finished = self._clock()
        log.info("Job %s done.", job.id)

    def _fail(self, job: Job, message: str, kind: str) -> None:
        job.status = JobStatus.FAILED
        # Messages can name an uploaded file: mask personal-data shapes like the report does.
        job.error = scrub_text(message or GENERIC_FAILURE)[:MAX_ERROR_CHARS]
        job.finished = self._clock()
        log.warning("Job %s failed (%s).", job.id, kind)


async def sweep_forever(sessions: SessionStore, jobs: JobStore, interval_s: float) -> None:
    """Ends expired sessions and deletes expired jobs every `interval_s` seconds."""
    while True:
        await asyncio.sleep(interval_s)
        try:
            ended = sessions.purge_expired()
            removed = jobs.purge_expired()
        except Exception as exc:  # noqa: BLE001 - keep sweeping; log only the class
            log.warning("Cleanup failed (%s).", type(exc).__name__)
            continue
        if ended or removed:
            log.info("Cleanup: ended %d sessions, deleted %d jobs.", ended, removed)
