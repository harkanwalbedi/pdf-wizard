"""Tests of the web UI: sign-in, uploads, jobs, and results (no network).

The fill pipeline is replaced by a stub that returns a hand-built, masked
report, except in the one integration test at the end, which runs the real
pipeline with a FakeMapper. A fake clock drives session, job, and rate-limit
timing. test_web_security.py imports the helpers defined here.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import threading
import time
import warnings
from collections.abc import Callable, Iterator, Sequence
from types import SimpleNamespace
from typing import Any

import pytest
from markupsafe import escape
from starlette.exceptions import StarletteDeprecationWarning

with warnings.catch_warnings():
    # Starlette 1.x recommends httpx2 for its test client; httpx is what's installed.
    warnings.simplefilter("ignore", StarletteDeprecationWarning)
    from fastapi.testclient import TestClient

import pdf_factory as pf
from pdf_wizard.config import Limits, Settings
from pdf_wizard.errors import InvalidInputError
from pdf_wizard.mapping.fake import FakeMapper
from pdf_wizard.models import (
    FieldResult,
    FieldStatus,
    FieldType,
    FillReport,
    RejectReason,
    SourceKind,
    SourceSummary,
)
from pdf_wizard.pipeline import FillOutcome
from pdf_wizard.web import create_app
from pdf_wizard.web.jobs import (
    GENERIC_FAILURE,
    TIMEOUT_FAILURE,
    JobPayload,
    JobRunner,
    JobStatus,
    JobStore,
    UploadItem,
)

# --------------------------------------------------------------------------- #
# Helpers (also used by test_web_security.py)
# --------------------------------------------------------------------------- #

TOKEN = "pdf-wizard-test-token-0123456789-abcdefghijkl"
SESSION_COOKIE = "__Host-pdf_wizard_session"
BASE_URL = "https://localhost"
PDF_BYTES = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<< >>\nendobj\ntrailer\n<< >>\n%%EOF\n"
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + bytes(32)
JPEG_BYTES = b"\xff\xd8\xff\xe0" + bytes(32)
JSON_BYTES = b'{"applicant": {"name": "Jane Q. Public", "state": "TX"}}'
FILLED_PDF = b"%PDF-1.7\n% filled by the stub\n%%EOF\n"
CSRF_RE = re.compile(r'name="csrf_token" value="([^"]+)"')


class FakeClock:
    """A monotonic clock the tests move by hand."""

    def __init__(self, start: float = 10_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_report(
    fields: Sequence[FieldResult] | None = None, *, warnings: Sequence[str] = (), masked: bool = True
) -> FillReport:
    """A report like the masked one `pipeline.fill_form` returns."""
    if fields is None:
        fields = [
            FieldResult(field_id="F1", name="full_name", label="Full name", type=FieldType.TEXT, page=1,
                        status=FieldStatus.FILLED, value="Jane Q. Public", source_id="S1",
                        evidence="applicant.name", confidence=0.97),
            FieldResult(field_id="F2", name="ssn", label="SSN", type=FieldType.TEXT, page=1,
                        status=FieldStatus.FILLED, value="***-**-6789", source_id="S1",
                        evidence="[hidden: sensitive]", confidence=0.99, sensitive=True),
            FieldResult(field_id="F3", name="email", label="Email address", type=FieldType.TEXT, page=1,
                        status=FieldStatus.NO_DATA),
            FieldResult(field_id="F4", name="state", label="State", type=FieldType.CHOICE, page=2,
                        status=FieldStatus.REJECTED, value="Ohio", source_id="S1", evidence="applicant.state",
                        confidence=0.9, reason=RejectReason.INVALID_VALUE,
                        detail="The value is not one of the field's options."),
            FieldResult(field_id="F5", name="office_use", label="Office use only", type=FieldType.TEXT, page=2,
                        status=FieldStatus.NOT_WRITABLE),
        ]
    counts = {str(status): 0 for status in FieldStatus}
    for result in fields:
        counts[str(result.status)] += 1
    return FillReport(
        tool_version="0.1.0",
        created_at="2026-10-06T12:00:00Z",
        form_name="form.pdf",
        form_sha256="ab" * 32,
        page_count=2,
        model="fake-model",
        prompt_version="fake",
        masked=masked,
        counts=counts,
        fields=list(fields),
        sources=[
            SourceSummary(id="S0", kind=SourceKind.SYSTEM, name="PDF Wizard"),
            SourceSummary(id="S1", kind=SourceKind.STRUCTURED, name="applicant.json"),
        ],
        warnings=list(warnings),
        timings_ms={"total": 1234},
    )


class StubFill:
    """Stands in for `pipeline.fill_form` and records every call."""

    def __init__(
        self,
        outcome: FillOutcome | None = None,
        *,
        error: Exception | None = None,
        gate: threading.Event | None = None,
    ) -> None:
        self.outcome = outcome or FillOutcome(pdf=FILLED_PDF, report=make_report())
        self.error = error
        self.gate = gate  # when set, each call waits for it
        self.calls: list[SimpleNamespace] = []

    def __call__(self, form_bytes, inputs, *, mapper, settings, options, form_name):  # noqa: ANN001, ANN204
        self.calls.append(SimpleNamespace(form=form_bytes, inputs=list(inputs), mapper=mapper,
                                          settings=settings, options=options, form_name=form_name))
        if self.gate is not None:
            self.gate.wait(timeout=10)
        if self.error is not None:
            raise self.error
        return self.outcome


def build_app(
    *,
    fill_fn: Callable[..., FillOutcome] | None,
    clock: FakeClock | None = None,
    settings: Settings | None = None,
    mapper_factory: Callable[[], Any] = FakeMapper,
):
    return create_app(
        settings or Settings(),
        login_token=TOKEN,
        mapper_factory=mapper_factory,
        fill_fn=fill_fn,
        clock=clock or FakeClock(),
    )


def open_client(app, **kwargs: Any) -> TestClient:
    """A client for https://localhost, so the Secure session cookie is sent back."""
    return TestClient(app, base_url=BASE_URL, **kwargs)


def csrf_from(html: str) -> str:
    match = CSRF_RE.search(html)
    assert match, "no CSRF token in the page"
    return match.group(1)


def esc(text: str) -> str:
    """`text` as it appears in an autoescaped page."""
    return str(escape(text))


def sign_in(client: TestClient) -> str:
    """Signs in and returns the session's CSRF token."""
    page = client.get("/login")
    assert page.status_code == 200
    response = client.post("/login", data={"csrf_token": csrf_from(page.text), "token": TOKEN},
                           follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/"
    index = client.get("/")
    assert index.status_code == 200
    return csrf_from(index.text)


def upload(
    client: TestClient,
    csrf: str | None,
    *,
    form: tuple[str, bytes] | None = ("form.pdf", PDF_BYTES),
    data: Sequence[tuple[str, bytes]] = (("applicant.json", JSON_BYTES),),
    docs: Sequence[tuple[str, bytes]] = (),
    references: Sequence[tuple[str, bytes]] = (),
    notes: str | None = None,
    row: str | None = None,
    flatten: bool = False,
    headers: dict[str, str] | None = None,
):
    files: list[tuple[str, tuple[str, bytes, str]]] = []
    if form is not None:
        files.append(("form", (form[0], form[1], "application/pdf")))
    for field, items in (("data", data), ("docs", docs), ("references", references)):
        files.extend((field, (name, content, "application/octet-stream")) for name, content in items)
    fields: dict[str, str] = {}
    if csrf is not None:
        fields["csrf_token"] = csrf
    if notes is not None:
        fields["notes"] = notes
    if row is not None:
        fields["row"] = row
    if flatten:
        fields["flatten"] = "1"
    return client.post("/jobs", data=fields, files=files, headers=headers, follow_redirects=False)


def multipart(parts: Sequence[tuple[str, str | None, bytes]], boundary: str = "pdfwizardtestboundary"):
    """A hand-built multipart body: (field name, file name or None for a text field, content)."""
    chunks: list[bytes] = []
    for name, filename, content in parts:
        disposition = f'form-data; name="{name}"'
        if filename is not None:
            disposition += f'; filename="{filename}"'
        head = f"--{boundary}\r\nContent-Disposition: {disposition}\r\n"
        if filename is not None:
            head += "Content-Type: application/octet-stream\r\n"
        chunks += [head.encode(), b"\r\n", content, b"\r\n"]
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), {"content-type": f"multipart/form-data; boundary={boundary}"}


def job_id_of(response) -> str:
    assert response.status_code == 303, response.text[:500]
    location = response.headers["location"]
    assert location.startswith("/jobs/")
    return location.removeprefix("/jobs/")


def wait_for_job(client: TestClient, job_id: str, *, timeout: float = 15.0) -> dict:
    """Polls the status API until the job is done or failed."""
    deadline = time.monotonic() + timeout
    delay = 0.005
    while True:
        response = client.get(f"/jobs/{job_id}/status")
        assert response.status_code == 200, response.text
        body = response.json()
        if body["status"] in ("done", "failed"):
            return body
        assert time.monotonic() < deadline, f"job is still {body['status']}"
        time.sleep(delay)
        delay = min(delay * 2, 0.25)


def cookie_payload(value: str) -> dict:
    """The JSON stored in Starlette's signed (not encrypted) session cookie."""
    data = value.split(".", 1)[0]
    return json.loads(base64.b64decode(data + "=" * (-len(data) % 4)))


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def stub() -> StubFill:
    return StubFill()


@pytest.fixture
def app(clock: FakeClock, stub: StubFill):
    return build_app(fill_fn=stub, clock=clock)


@pytest.fixture
def client(app) -> Iterator[TestClient]:
    with open_client(app) as test_client:
        yield test_client


# --------------------------------------------------------------------------- #
# Sign-in
# --------------------------------------------------------------------------- #


def test_sign_in_flow(client: TestClient):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/login"

    page = client.get("/login")
    assert page.status_code == 200
    assert 'type="password"' in page.text and 'autocomplete="off"' in page.text
    assert "printed in the terminal" in page.text
    assert 'id="logout-form"' not in page.text

    csrf = sign_in(client)
    index = client.get("/")
    assert "<h1>Fill a PDF form</h1>" in index.text
    assert 'enctype="multipart/form-data"' in index.text
    assert f'name="csrf_token" value="{csrf}"' in index.text
    assert 'id="logout-form"' in index.text
    assert "No jobs yet" in index.text

    # Signed in: the sign-in page sends you home.
    response = client.get("/login", follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/"


def test_sign_in_failure_is_generic(client: TestClient):
    wrong = "not-the-right-token-but-long-enough-0000"
    page = client.get("/login")
    response = client.post("/login", data={"csrf_token": csrf_from(page.text), "token": wrong},
                           follow_redirects=False)
    assert response.status_code == 401
    assert esc("That token didn't work") in response.text
    assert wrong not in response.text and TOKEN not in response.text
    assert client.get("/", follow_redirects=False).status_code == 303  # still signed out

    for bad in ("", TOKEN[:-1], TOKEN + "x", TOKEN.upper()):
        response = client.post("/login", data={"csrf_token": csrf_from(page.text), "token": bad},
                               follow_redirects=False)
        assert response.status_code == 401, bad


def test_sign_in_ignores_surrounding_whitespace(client: TestClient):
    page = client.get("/login")
    response = client.post("/login", data={"csrf_token": csrf_from(page.text), "token": f"  {TOKEN}\n"},
                           follow_redirects=False)
    assert response.status_code == 303
    assert client.get("/").status_code == 200


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #


def test_job_flow_to_done(client: TestClient, stub: StubFill):
    csrf = sign_in(client)
    response = upload(
        client,
        csrf,
        data=[("applicant.csv", b"name,state\nJane,TX\nJohn,CA\n")],
        docs=[("letter.txt", b"Dear Jane"), ("scan.png", PNG_BYTES), ("photo.JPG", JPEG_BYTES),
              ("statement.pdf", PDF_BYTES), ("mail.eml", b"From: a@example.com\r\n\r\nHi")],
        references=[("last-year.pdf", PDF_BYTES)],
        notes="Use my work address.\r\nThanks!",
        row="2",
        flatten=True,
    )
    job_id = job_id_of(response)
    status = wait_for_job(client, job_id)
    assert status == {"status": "done", "counts": {"filled": 2, "no_data": 1, "rejected": 1, "not_writable": 1}}

    # What the pipeline was given.
    [call] = stub.calls
    assert call.form == PDF_BYTES and call.form_name == "form.pdf"
    assert call.options.flatten is True and call.settings is not None
    assert isinstance(call.mapper, FakeMapper)
    assert [(i.role, i.name, i.row) for i in call.inputs] == [
        ("data", "applicant.csv", 2),
        ("doc", "letter.txt", None),
        ("doc", "scan.png", None),
        ("doc", "photo.JPG", None),
        ("doc", "statement.pdf", None),
        ("doc", "mail.eml", None),
        ("reference", "last-year.pdf", None),
        ("note", "", None),
    ]
    assert call.inputs[-1].content == "Use my work address.\nThanks!".encode()
    assert call.inputs[2].content == PNG_BYTES

    page = client.get(f"/jobs/{job_id}")
    assert page.status_code == 200
    html = page.text
    assert "<title>form.pdf · PDF Wizard</title>" in html
    assert f'href="/jobs/{job_id}/download"' in html
    for text in ("Filled", "No data", "Rejected", "Not writable", "Full name", "Jane Q. Public",
                 "***-**-6789", "[hidden: sensitive]", "Invalid value", "The value is not one of the field",
                 "applicant.json", "97%", "fake-model", "1.2 s", "Sensitive values and their evidence are masked"):
        assert text in html, text
    assert 'data-status="rejected"' in html and "<meter" in html
    assert "data-status-url" not in html  # no polling once done

    index = client.get("/").text
    assert f'href="/jobs/{job_id}"' in index and "badge-done" in index

    download = client.get(f"/jobs/{job_id}/download")
    assert download.status_code == 200
    assert download.content == FILLED_PDF
    assert download.headers["content-type"] == "application/pdf"
    assert download.headers["content-disposition"] == 'attachment; filename="form.filled.pdf"'
    assert download.headers["x-content-type-options"] == "nosniff"
    assert download.headers["cache-control"] == "no-store"


def test_note_only_upload_is_accepted(client: TestClient, stub: StubFill):
    csrf = sign_in(client)
    job_id = job_id_of(upload(client, csrf, data=(), notes="My name is Jane."))
    assert wait_for_job(client, job_id)["status"] == "done"
    assert [(i.role, i.content) for i in stub.calls[0].inputs] == [("note", b"My name is Jane.")]
    assert stub.calls[0].options.flatten is False


def test_download_name_is_sanitized(client: TestClient):
    # (Browsers and httpx send '"' in multipart file names as %22, so it isn't used here.)
    csrf = sign_in(client)
    job_id = job_id_of(upload(client, csrf, form=("../../etc/my tax form (2026)!.pdf", PDF_BYTES)))
    wait_for_job(client, job_id)
    download = client.get(f"/jobs/{job_id}/download")
    assert download.headers["content-disposition"] == 'attachment; filename="my_tax_form_2026.filled.pdf"'
    assert "../" not in client.get(f"/jobs/{job_id}").text


def test_form_name_shown_with_personal_data_masked(client: TestClient):
    csrf = sign_in(client)
    job_id = job_id_of(upload(client, csrf, form=("jane.doe@example.com 123-45-6789.pdf", PDF_BYTES)))
    wait_for_job(client, job_id)
    for html in (client.get(f"/jobs/{job_id}").text, client.get("/").text):
        assert "jane.doe@example.com" not in html and "123-45-6789" not in html
        assert "6789.pdf" in html


def test_empty_file_inputs_are_ignored(client: TestClient, stub: StubFill):
    csrf = sign_in(client)
    body, headers = multipart([
        ("csrf_token", None, csrf.encode()),
        ("form", "form.pdf", PDF_BYTES),
        ("data", "applicant.json", JSON_BYTES),
        ("docs", "", b""),  # what a browser sends for an empty <input type="file">
        ("references", "", b""),
        ("notes", None, b""),
        ("row", None, b""),
    ])
    response = client.post("/jobs", content=body, headers=headers, follow_redirects=False)
    wait_for_job(client, job_id_of(response))
    assert [i.role for i in stub.calls[0].inputs] == ["data"]


def test_job_page_while_running(clock: FakeClock):
    gate = threading.Event()
    app = build_app(fill_fn=StubFill(gate=gate), clock=clock)
    with open_client(app) as client:
        try:
            csrf = sign_in(client)
            job_id = job_id_of(upload(client, csrf))
            status = client.get(f"/jobs/{job_id}/status").json()
            assert status["status"] in ("queued", "running") and status["counts"] == {}
            page = client.get(f"/jobs/{job_id}").text
            assert f'data-status-url="/jobs/{job_id}/status"' in page
            assert 'class="spinner"' in page and 'http-equiv="refresh"' in page
            assert "/download" not in page
            assert client.get(f"/jobs/{job_id}/download").status_code == 404
            assert "badge-running" in client.get("/").text or "badge-queued" in client.get("/").text
        finally:
            gate.set()
        assert wait_for_job(client, job_id)["status"] == "done"


def test_failed_job_shows_only_the_user_message(clock: FakeClock):
    app = build_app(fill_fn=StubFill(error=InvalidInputError("'data.json' is not valid JSON.")), clock=clock)
    with open_client(app) as client:
        csrf = sign_in(client)
        job_id = job_id_of(upload(client, csrf))
        assert wait_for_job(client, job_id) == {"status": "failed", "counts": {}}
        page = client.get(f"/jobs/{job_id}").text
        assert "This form couldn't be filled" in page  # literal template text
        assert esc("'data.json' is not valid JSON.") in page  # the error's user_message, escaped
        assert "/download" not in page and "<table" not in page
        assert client.get(f"/jobs/{job_id}/download").status_code == 404
        assert "badge-failed" in client.get("/").text


def test_unexpected_errors_show_a_generic_message(clock: FakeClock, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG)
    secret = "Traceback detail /home/jane/ssn 123-45-6789"
    app = build_app(fill_fn=StubFill(error=RuntimeError(secret)), clock=clock)
    with open_client(app) as client:
        csrf = sign_in(client)
        job_id = job_id_of(upload(client, csrf))
        wait_for_job(client, job_id)
        page = client.get(f"/jobs/{job_id}").text
    assert GENERIC_FAILURE in page and secret not in page
    assert "RuntimeError" in caplog.text and secret not in caplog.text


def test_unmasked_report_is_never_shown(clock: FakeClock):
    raw = make_report([FieldResult(field_id="F1", name="ssn", type=FieldType.TEXT, status=FieldStatus.FILLED,
                                   value="123-45-6789", sensitive=True)], masked=False)
    app = build_app(fill_fn=StubFill(FillOutcome(pdf=FILLED_PDF, report=raw)), clock=clock)
    with open_client(app) as client:
        csrf = sign_in(client)
        job_id = job_id_of(upload(client, csrf))
        assert wait_for_job(client, job_id)["status"] == "failed"
        page = client.get(f"/jobs/{job_id}").text
    assert "123-45-6789" not in page and GENERIC_FAILURE in page


def test_values_from_documents_are_escaped(clock: FakeClock):
    script = "<script>alert(1)</script>"
    report = make_report(
        [FieldResult(field_id="F1", name=f"name{script}", label=f"Label {script}", type=FieldType.TEXT,
                     page=1, status=FieldStatus.FILLED, value=f'"><img src=x onerror=alert(1)>{script}',
                     source_id="S1", evidence=f"evidence {script}" * 10, confidence=0.9,
                     notes=[f"note {script}"])],
        warnings=[f"warning {script}"],
    )
    app = build_app(fill_fn=StubFill(FillOutcome(pdf=FILLED_PDF, report=report)), clock=clock)
    with open_client(app) as client:
        csrf = sign_in(client)
        job_id = job_id_of(upload(client, csrf, form=(f"{script}.pdf", PDF_BYTES)))
        wait_for_job(client, job_id)
        pages = [client.get(f"/jobs/{job_id}").text, client.get("/").text]
    for html in pages:
        assert script not in html and "<img src=x" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in pages[0]
    assert "&#34;&gt;&lt;img src=x onerror=alert(1)&gt;" in pages[0]
    assert "Show all" not in pages[0]  # the label for <details> comes from CSS, not markup
    assert '<details class="more evidence">' in pages[0]


def test_delete_job(client: TestClient):
    csrf = sign_in(client)
    job_id = job_id_of(upload(client, csrf))
    wait_for_job(client, job_id)
    response = client.post(f"/jobs/{job_id}/delete", data={"csrf_token": csrf}, follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/"
    assert client.get(f"/jobs/{job_id}").status_code == 404
    assert client.get(f"/jobs/{job_id}/status").status_code == 404
    assert f"/jobs/{job_id}" not in client.get("/").text
    # Deleting again: same answer as for any unknown job.
    response = client.post(f"/jobs/{job_id}/delete", data={"csrf_token": csrf}, follow_redirects=False)
    assert response.status_code == 404


def test_jobs_of_other_sessions_are_not_found(app):
    with open_client(app) as alice, open_client(app) as mallory:
        alice_csrf = sign_in(alice)
        mallory_csrf = sign_in(mallory)
        job_id = job_id_of(upload(alice, alice_csrf))
        wait_for_job(alice, job_id)

        for path in (f"/jobs/{job_id}", f"/jobs/{job_id}/download"):
            response = mallory.get(path)
            assert response.status_code == 404 and "Not found" in response.text
        response = mallory.get(f"/jobs/{job_id}/status")
        unknown = mallory.get(f"/jobs/{'A' * len(job_id)}/status")
        assert response.status_code == unknown.status_code == 404
        assert response.json() == unknown.json()  # "not yours" looks exactly like "doesn't exist"
        response = mallory.post(f"/jobs/{job_id}/delete", data={"csrf_token": mallory_csrf})
        assert response.status_code == 404
        assert f"/jobs/{job_id}" not in mallory.get("/").text

        assert alice.get(f"/jobs/{job_id}").status_code == 200  # untouched


def test_unknown_and_malformed_job_ids(client: TestClient):
    csrf = sign_in(client)
    for job_id in ("A" * 22, "short", "a" * 65, "..%2F..%2Fetc", "%3Cscript%3E"):
        assert client.get(f"/jobs/{job_id}").status_code == 404
        assert client.get(f"/jobs/{job_id}/status").status_code == 404
        assert client.post(f"/jobs/{job_id}/delete", data={"csrf_token": csrf}).status_code == 404


def test_recent_jobs_are_listed_newest_first(client: TestClient, clock: FakeClock):
    csrf = sign_in(client)
    first = job_id_of(upload(client, csrf, form=("first.pdf", PDF_BYTES)))
    clock.advance(120)
    second = job_id_of(upload(client, csrf, form=("second.pdf", PDF_BYTES)))
    wait_for_job(client, first)
    wait_for_job(client, second)
    index = client.get("/").text
    assert index.index(second) < index.index(first)
    assert "2 min ago" in index and "just now" in index and "2 of 20" in index


# --------------------------------------------------------------------------- #
# Upload validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"form": None}, "Choose the PDF form you want to fill."),
        ({"form": ("form.pdf", PNG_BYTES)}, "The form to fill must be a PDF file (.pdf)."),
        ({"form": ("form.exe", PDF_BYTES)}, "The form to fill must be a PDF file (.pdf)."),
        ({"form": ("form.pdf", b"")}, "The form to fill must be a PDF file (.pdf)."),
        ({"docs": [("setup.exe", b"MZ\x90\x00")]}, "Documents must be PDF (.pdf), PNG (.png)"),
        ({"docs": [("photo.png", PDF_BYTES)]}, "Documents must be PDF (.pdf), PNG (.png)"),
        ({"docs": [("notes.txt", b"\xff\xfe not utf-8")]}, "Documents must be PDF (.pdf), PNG (.png)"),
        ({"docs": [("noext", b"plain text")]}, "Documents must be PDF (.pdf), PNG (.png)"),
        ({"data": [("data.json", b'{"a": 1}\x00')]}, "Data files must be JSON (.json) or CSV (.csv)"),
        ({"data": [("data.xlsx", b"PK\x03\x04")]}, "Data files must be JSON (.json) or CSV (.csv)"),
        ({"references": [("old.pdf", PNG_BYTES)]}, "Previously filled forms must be PDF files (.pdf)."),
        ({"data": ()}, "Add at least one data file, document, previously filled form, or note."),
        ({"data": (), "notes": "   \r\n  "}, "Add at least one data file, document, previously filled form, or note."),
        ({"row": "abc"}, "The CSV row must be a whole number from 1 to 100,000."),
        ({"row": "0"}, "The CSV row must be a whole number from 1 to 100,000."),
        ({"row": "100001"}, "The CSV row must be a whole number from 1 to 100,000."),
        ({"row": "-1"}, "The CSV row must be a whole number from 1 to 100,000."),
        ({"notes": "x" * 5_001}, "The note can be up to 5,000 characters."),
    ],
)
def test_upload_validation(client: TestClient, stub: StubFill, kwargs: dict, message: str):
    csrf = sign_in(client)
    response = upload(client, csrf, **kwargs)
    assert response.status_code == 400
    assert message.replace("'", "&#39;") in response.text
    assert 'id="upload-form"' in response.text  # the upload page again, with the error
    assert stub.calls == [] and "No jobs yet" in response.text


def test_upload_rejects_two_forms(client: TestClient):
    csrf = sign_in(client)
    body, headers = multipart([("csrf_token", None, csrf.encode()), ("form", "a.pdf", PDF_BYTES),
                               ("form", "b.pdf", PDF_BYTES), ("notes", None, b"hi")])
    response = client.post("/jobs", content=body, headers=headers)
    assert response.status_code == 400 and "Choose only one form to fill." in response.text


def test_upload_size_and_count_limits(clock: FakeClock):
    settings = Settings(limits=Limits(max_file_bytes=2_000, max_total_bytes=5_000, max_source_files=2))
    stub = StubFill()
    app = build_app(fill_fn=stub, clock=clock, settings=settings)
    with open_client(app) as client:
        csrf = sign_in(client)
        big = PDF_BYTES + bytes(2_000)
        response = upload(client, csrf, form=("form.pdf", big))
        assert response.status_code == 413 and "Each file must be" in response.text
        response = upload(client, csrf, docs=[("big.txt", b"a" * 2_001)])
        assert response.status_code == 413
        response = upload(client, csrf, data=(), docs=[("a.txt", b"a" * 1_900), ("b.txt", b"b" * 1_900)],
                          form=("form.pdf", PDF_BYTES + bytes(1_500)))
        assert response.status_code == 413 and "All files together must be" in response.text
        response = upload(client, csrf, docs=[("a.txt", b"a"), ("b.txt", b"b")])  # 3 files > 2
        assert response.status_code == 400 and "Add up to 2 files besides the form." in response.text
        # Far too many files: the multipart parser stops early.
        response = upload(client, csrf, docs=[(f"{i}.txt", b"x") for i in range(12)])
        assert response.status_code == 400 and "That upload couldn" in response.text
        assert stub.calls == []


def test_oversized_note_field_is_rejected_by_the_parser(client: TestClient, stub: StubFill):
    csrf = sign_in(client)
    response = upload(client, csrf, notes="x" * (65 * 1024))
    assert response.status_code == 400
    assert "That upload couldn" in response.text and "exceeded" not in response.text
    assert stub.calls == []


# --------------------------------------------------------------------------- #
# Caps and expiry
# --------------------------------------------------------------------------- #


def test_job_cap_per_session(client: TestClient, clock: FakeClock):
    csrf = sign_in(client)
    for number in range(20):
        if number and number % 6 == 0:
            clock.advance(61)  # stay under the upload rate limit
        assert upload(client, csrf).status_code == 303
    clock.advance(61)
    response = upload(client, csrf)
    assert response.status_code == 429
    assert "You can keep up to 20 jobs at a time." in response.text


def test_job_cap_in_total(client: TestClient, app):
    csrf = sign_in(client)
    for number in range(100):
        app.state.web.jobs.create(f"another-session-{number // 20}", "other.pdf")
    response = upload(client, csrf)
    assert response.status_code == 503
    assert "PDF Wizard is busy with other jobs." in response.text


def test_jobs_expire_after_an_hour(client: TestClient, clock: FakeClock):
    csrf = sign_in(client)
    job_id = job_id_of(upload(client, csrf))
    wait_for_job(client, job_id)
    for _ in range(3):
        clock.advance(20 * 60)  # keep the session alive
        client.get("/")
    assert client.get(f"/jobs/{job_id}").status_code == 404
    assert len(client.app.state.web.jobs) == 0


def test_idle_session_expires(client: TestClient, clock: FakeClock):
    csrf = sign_in(client)
    job_id = job_id_of(upload(client, csrf))
    wait_for_job(client, job_id)
    clock.advance(29 * 60)
    assert client.get("/").status_code == 200  # activity refreshes the idle timer
    clock.advance(30 * 60 + 1)
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/login"
    web = client.app.state.web
    assert len(web.sessions) == 0 and len(web.jobs) == 0  # the session's jobs went with it
    assert client.get(f"/jobs/{job_id}/status").status_code == 401


def test_absolute_session_lifetime(client: TestClient, clock: FakeClock):
    sign_in(client)
    for _ in range(16):  # 16 x 29 min = 7 h 44 min
        clock.advance(29 * 60)
        assert client.get("/", follow_redirects=False).status_code == 200
    clock.advance(29 * 60)  # 8 h 13 min after sign-in
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/login"


def test_logout_ends_the_session(client: TestClient):
    csrf = sign_in(client)
    job_id = job_id_of(upload(client, csrf))
    wait_for_job(client, job_id)
    cookie = client.cookies.get(SESSION_COOKIE)

    response = client.post("/logout", data={"csrf_token": csrf}, follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/login"
    assert response.headers["clear-site-data"] == '"cache", "cookies", "storage"'
    web = client.app.state.web
    assert len(web.sessions) == 0 and len(web.jobs) == 0
    assert client.get("/", follow_redirects=False).status_code == 303

    # The old cookie is useless now.
    client.cookies.clear()
    response = client.get(f"/jobs/{job_id}", headers={"cookie": f"{SESSION_COOKIE}={cookie}"},
                          follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/login"


def test_sweeper_purges_expired_sessions_and_jobs():
    from pdf_wizard.web.jobs import sweep_forever
    from pdf_wizard.web.security import SessionStore

    clock = FakeClock()
    jobs = JobStore(clock=clock, ttl_s=3600, max_per_owner=20, max_total=100)
    sessions = SessionStore(clock=clock, idle_s=1800, absolute_s=8 * 3600, max_sessions=20,
                            on_end=jobs.delete_owner)
    record = sessions.create()
    jobs.create(record.sid, "a.pdf")
    jobs.create("someone-else", "b.pdf")
    clock.advance(1801)

    async def run_briefly() -> None:
        task = asyncio.create_task(sweep_forever(sessions, jobs, 0.01))
        await asyncio.sleep(0.05)
        task.cancel()

    asyncio.run(run_briefly())
    assert len(sessions) == 0
    assert len(jobs) == 1  # only the other owner's job is left; it expires later
    clock.advance(1800)
    assert jobs.purge_expired() == 1 and len(jobs) == 0


def test_runner_stops_slow_jobs():
    clock = FakeClock()
    store = JobStore(clock=clock, ttl_s=3600, max_per_owner=20, max_total=100)

    def slow_fill(*args: Any, **kwargs: Any) -> FillOutcome:
        time.sleep(0.3)
        return FillOutcome(pdf=FILLED_PDF, report=make_report())

    runner = JobRunner(store, settings=Settings(), mapper_factory=FakeMapper, fill_fn=slow_fill,
                       max_concurrent=1, timeout_s=0.05, clock=clock)
    job = store.create("owner", "form.pdf")
    payload = JobPayload(form=PDF_BYTES, form_name="form.pdf",
                         inputs=(UploadItem(role="note", name="", content=b"hi"),))

    async def run() -> None:
        runner.start()
        runner.submit(job, payload)
        await asyncio.wait_for(asyncio.gather(*runner._tasks), timeout=5)

    asyncio.run(run())
    assert job.status is JobStatus.FAILED and job.error == TIMEOUT_FAILURE
    assert job.pdf is None and job.report is None


# --------------------------------------------------------------------------- #
# Integration with the real pipeline
# --------------------------------------------------------------------------- #


def test_real_pipeline_end_to_end(clock: FakeClock):
    pytest.importorskip("pdf_wizard.pipeline")
    from pdf_wizard.mapping.fake import proposal
    from pdf_wizard.pdf.inspector import open_form

    ssn = "123-45-6789"
    record = {"applicant": {"name": "Jane Q. Public", "ssn": ssn, "state": "TX"}}

    def propose(request):  # noqa: ANN001, ANN202
        ids = {field.name: field.id for field in request.fields}
        return [
            proposal(ids["full_name"], "Jane Q. Public", evidence="applicant.name"),
            proposal(ids["ssn"], ssn, evidence="applicant.ssn", confidence=0.99, sensitive=True),
            proposal(ids["state"], "TX", evidence="applicant.state"),
            proposal(ids["email"], "jane@example.com", evidence="jane@example.com"),  # not in the data
        ]

    app = build_app(fill_fn=None, clock=clock, mapper_factory=lambda: FakeMapper(fn=propose))
    with open_client(app) as client:
        csrf = sign_in(client)
        response = upload(client, csrf, form=("sample.pdf", pf.sample_form()),
                          data=[("applicant.json", json.dumps(record).encode())])
        job_id = job_id_of(response)
        status = wait_for_job(client, job_id, timeout=60)
        page = client.get(f"/jobs/{job_id}").text
        assert status["status"] == "done", page
        assert status["counts"]["filled"] == 3 and status["counts"]["rejected"] == 1
        assert "Jane Q. Public" in page
        assert "*****6789" in page  # the pipeline normalizes the SSN to digits, then masks it
        assert ssn not in page and "123456789" not in page and "jane@example.com" not in page
        assert "Evidence not found" in page and "[hidden: sensitive]" in page

        download = client.get(f"/jobs/{job_id}/download")
        assert download.status_code == 200
        assert download.headers["content-disposition"] == 'attachment; filename="sample.filled.pdf"'
        fields = {f.name: f for f in open_form(download.content, for_filling=False).info.fields}
        assert fields["full_name"].current_value == "Jane Q. Public"
        assert fields["ssn"].current_value == "123456789"  # the PDF has the full value
        assert fields["email"].current_value is None
