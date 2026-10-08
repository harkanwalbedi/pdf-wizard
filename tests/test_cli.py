"""CLI tests with a fake mapper (no network)."""

from __future__ import annotations

import io
import json
import logging
import stat

import pytest

import pdf_factory as pf
from pdf_wizard.cli import run
from pdf_wizard.errors import MappingError
from pdf_wizard.mapping.base import MappingRequest
from pdf_wizard.mapping.fake import FakeMapper, proposal

SSN = "123-45-6789"


def name_mapper(**values):
    """Proposes values by field name, citing JSON key paths of the same name in S1."""

    def fn(request: MappingRequest):
        ids = {f.name: f.id for f in request.fields}
        return [
            proposal(ids[name], value, source_id="S1", evidence=name, sensitive=name == "ssn")
            for name, value in values.items()
        ]

    return lambda settings: FakeMapper(fn=fn)


@pytest.fixture
def files(tmp_path):
    form = tmp_path / "form.pdf"
    form.write_bytes(pf.sample_form())
    data = tmp_path / "data.json"
    data.write_text(json.dumps({"full_name": "Jane Q. Public", "ssn": SSN}))
    return tmp_path, form, data


def cli(argv, mapper_factory=None):
    out, err = io.StringIO(), io.StringIO()
    code = run([str(a) for a in argv], mapper_factory=mapper_factory, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_fill_writes_private_outputs_and_masks_the_report(files):
    tmp, form, data = files
    out_pdf, report = tmp / "out.pdf", tmp / "report.json"
    code, out, err = cli(
        ["fill", form, "-o", out_pdf, "--data", data, "--report", report],
        name_mapper(full_name="Jane Q. Public", ssn=SSN),
    )
    assert code == 0, err
    assert out_pdf.read_bytes().startswith(b"%PDF")
    assert stat.S_IMODE(out_pdf.stat().st_mode) == 0o600
    assert stat.S_IMODE(report.stat().st_mode) == 0o600
    payload = json.loads(report.read_text())
    assert payload["masked"] is True and payload["counts"]["filled"] == 2
    assert SSN not in report.read_text() and "123456789" not in report.read_text()
    assert "filled 2 of" in out and "Jane Q. Public" in out
    assert SSN not in out and "123456789" not in out  # the summary shows masked values
    assert not list(tmp.glob(".pdf-wizard-*"))  # no temp files left


def test_refuses_to_overwrite_without_force(files):
    tmp, form, data = files
    out_pdf = tmp / "out.pdf"
    out_pdf.write_bytes(b"existing")
    code, _, err = cli(["fill", form, "-o", out_pdf, "--data", data], name_mapper(full_name="Jane"))
    assert code == 2 and "already exists" in err
    assert out_pdf.read_bytes() == b"existing"
    code, _, _ = cli(["fill", form, "-o", out_pdf, "--data", data, "--force"], name_mapper(full_name="Jane"))
    assert code == 0 and out_pdf.read_bytes().startswith(b"%PDF")


def test_file_created_during_the_model_call_is_not_overwritten(files):
    tmp, form, data = files
    out_pdf = tmp / "out.pdf"

    def fn(request: MappingRequest):
        out_pdf.write_bytes(b"appeared meanwhile")  # e.g. written by another program
        ids = {f.name: f.id for f in request.fields}
        return [proposal(ids["full_name"], "Jane", source_id="S1", evidence="full_name")]

    code, _, err = cli(["fill", form, "-o", out_pdf, "--data", data], lambda settings: FakeMapper(fn=fn))
    assert code == 2 and "already exists" in err
    assert out_pdf.read_bytes() == b"appeared meanwhile"
    assert not list(tmp.glob(".pdf-wizard-*"))  # no temp files left


def test_never_overwrites_an_input(files):
    tmp, form, data = files
    original = form.read_bytes()
    code, _, err = cli(["fill", form, "-o", form, "--data", data, "--force"], name_mapper(full_name="Jane"))
    assert code == 2 and "input file" in err
    assert form.read_bytes() == original


def test_dry_run_writes_no_pdf(files):
    tmp, form, data = files
    code, out, _ = cli(["fill", form, "--dry-run", "--data", data], name_mapper(full_name="Jane"))
    assert code == 0 and "would fill 1 of" in out
    assert list(tmp.glob("*.pdf")) == [form]


@pytest.mark.parametrize(
    ("argv_tail", "expected_code"),
    [
        (["--data", "missing.json"], 2),  # unreadable input
        ([], 2),  # no sources
    ],
)
def test_invalid_input_exit_codes(files, argv_tail, expected_code):
    tmp, form, _ = files
    tail = [tmp / a if a.endswith(".json") else a for a in argv_tail]
    code, _, err = cli(["fill", form, "-o", tmp / "out.pdf", *tail], name_mapper())
    assert code == expected_code and err.startswith("Error:")


def test_unsupported_form_exit_code(files):
    tmp, _, data = files
    plain = tmp / "plain.pdf"
    plain.write_bytes(pf.make_text_pdf(["no fields"]))
    code, _, err = cli(["fill", plain, "-o", tmp / "out.pdf", "--data", data], name_mapper())
    assert code == 3 and "no fillable" in err


def test_mapping_error_exit_code(files):
    tmp, form, data = files
    code, _, err = cli(
        ["fill", form, "-o", tmp / "out.pdf", "--data", data],
        lambda settings: FakeMapper(error=MappingError("The AI model returned an unusable response.")),
    )
    assert code == 4 and "unusable" in err
    assert not (tmp / "out.pdf").exists()


def test_missing_cloud_config_fails_fast(files, monkeypatch):
    tmp, form, data = files
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("PDF_WIZARD_BACKEND", raising=False)
    code, _, err = cli(["fill", form, "-o", tmp / "out.pdf", "--data", data])
    assert code == 4 and "GOOGLE_CLOUD_PROJECT" in err


def test_inspect_table_and_json(files):
    tmp, form, _ = files
    code, out, _ = cli(["inspect", form])
    assert code == 0 and "full_name" in out and "Full name" in out and "employed (Employed)" in out

    prefilled = tmp / "prefilled.pdf"
    prefilled.write_bytes(pf.make_form([pf.Spec("text", "ssn", label="SSN", value=SSN)]))
    code, out, _ = cli(["inspect", prefilled, "--json"])
    payload = json.loads(out)
    assert code == 0 and payload["fields"][0]["name"] == "ssn"
    assert SSN not in out  # current values that look sensitive are masked


def test_terminal_control_characters_are_stripped(files):
    tmp, form, data = files
    data.write_text(json.dumps({"full_name": "Jane\x1b[31m Evil"}))
    code, out, _ = cli(["fill", form, "--dry-run", "--data", data], name_mapper(full_name="Jane\x1b[31m Evil"))
    assert "\x1b" not in out


def test_canary_ssn_never_reaches_logs(files, caplog, monkeypatch):
    """A canary SSN in every kind of input never shows up in log records, at any level or logger."""
    tmp, form, data = files
    letter = tmp / "letter.txt"
    letter.write_text(f"Applicant: Jane Q. Public\nSSN: {SSN}\n")
    scan = tmp / "scan.pdf"
    scan.write_bytes(pf.make_text_pdf([f"Social Security number {SSN}"]))
    email = tmp / "message.eml"
    email.write_text(
        "From: Jane <jane@example.com>\nTo: HR <hr@example.com>\nSubject: My details\n"
        "Date: Mon, 5 Oct 2026 10:00:00 +0000\nMessage-ID: <1@example.com>\nMIME-Version: 1.0\n"
        f"Content-Type: text/plain; charset=utf-8\n\nHi, my SSN is {SSN}.\n"
    )
    reference = tmp / "previous.pdf"
    reference.write_bytes(pf.make_form([pf.Spec("text", "ssn", label="SSN", value=SSN)]))
    monkeypatch.setattr("pdf_wizard.cli._setup_logging", lambda: None)  # keep global logging untouched
    caplog.set_level(logging.DEBUG)  # every record from every logger, before any redaction

    code, out, err = cli(
        ["fill", form, "-o", tmp / "out.pdf", "--report", tmp / "report.json", "--data", data,
         "--doc", letter, "--doc", scan, "--doc", email, "--reference", reference, "--note", f"My SSN is {SSN}"],
        name_mapper(full_name="Jane Q. Public", ssn=SSN),
    )

    assert code == 0, err
    assert caplog.records, "expected some log records"
    for secret in (SSN, "123456789", "Jane Q. Public", "jane@example.com"):
        assert secret not in caplog.text
        assert secret not in err
