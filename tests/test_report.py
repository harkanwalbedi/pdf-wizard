"""Tests for report: building the fill report and masking it."""

from __future__ import annotations

from datetime import datetime

from pdf_wizard import __version__
from pdf_wizard.models import (
    FieldResult,
    FieldStatus,
    FieldType,
    FormInfo,
    MappingResult,
    RejectReason,
    Source,
    SourceKind,
)
from pdf_wizard.report import HIDDEN_EVIDENCE, build_report, mask_report

SSN = "123-45-6789"
CARD = "4111 1111 1111 1111"


def form_info() -> FormInfo:
    return FormInfo(sha256="ab" * 32, page_count=2, fields=[])


def results() -> list[FieldResult]:
    return [
        FieldResult(
            field_id="F1",
            name="ssn",
            label="SSN",
            type=FieldType.TEXT,
            status=FieldStatus.FILLED,
            value=SSN,
            source_id="S1",
            evidence=f"SSN: {SSN}",
            confidence=0.97,
            sensitive=True,
        ),
        FieldResult(
            field_id="F2",
            name="name",
            type=FieldType.TEXT,
            status=FieldStatus.FILLED,
            value="Jane Doe",
            source_id="S1",
            evidence=f"Jane Doe, card {CARD}",
            confidence=0.9,
        ),
        FieldResult(
            field_id="F3",
            name="acct",
            type=FieldType.TEXT,
            status=FieldStatus.REJECTED,
            value="9876543210",
            source_id="S2",
            evidence="Account 9876543210",
            confidence=0.4,
            reason=RejectReason.LOW_CONFIDENCE,
            detail="Confidence 0.40 is below 0.70",
        ),
        FieldResult(field_id="F4", name="notes", type=FieldType.TEXT, status=FieldStatus.NO_DATA),
    ]


def sources() -> list[Source]:
    return [
        Source(id="S0", kind=SourceKind.SYSTEM, name="PDF Wizard", text="Today's date: 2026-10-06"),
        Source(id="S1", kind=SourceKind.TEXT, name=f"ssn {SSN}.txt", text="Jane Doe"),
        Source(id="S2", kind=SourceKind.IMAGE, name="scan.png", media_type="image/png", data=b"\x89PNG"),
    ]


def make_report(mapping: MappingResult | None = None):
    mapping = mapping or MappingResult(
        proposals=[], model="gemini-test", prompt_version="2026-10-06.1", usage={"input_tokens": 10, "total_tokens": 12}
    )
    return build_report(
        form_info=form_info(),
        form_name=f"form for {SSN}.pdf",
        results=results(),
        sources=sources(),
        mapping=mapping,
        flattened=True,
        dry_run=False,
        warnings=[f"Something about jane@example.com and {SSN}"],
        timings_ms={"mapping": 1200, "total": 1500},
    )


def test_build_report_fields() -> None:
    report = make_report()
    assert report.tool == "pdf-wizard"
    assert report.tool_version == __version__
    assert datetime.fromisoformat(report.created_at).utcoffset().total_seconds() == 0
    assert report.form_sha256 == "ab" * 32 and report.page_count == 2
    assert (report.model, report.prompt_version) == ("gemini-test", "2026-10-06.1")
    assert report.usage == {"input_tokens": 10, "total_tokens": 12}
    assert report.flattened and not report.dry_run and not report.masked
    assert report.timings_ms == {"mapping": 1200, "total": 1500}
    assert [f.field_id for f in report.fields] == ["F1", "F2", "F3", "F4"]


def test_counts_have_every_status() -> None:
    report = make_report()
    assert report.counts == {"filled": 2, "no_data": 1, "rejected": 1, "not_writable": 0}
    assert report.counts[FieldStatus.FILLED] == 2


def test_source_summaries() -> None:
    summaries = {s.id: s for s in make_report().sources}
    assert summaries["S0"].chars == len("Today's date: 2026-10-06")
    assert (summaries["S1"].chars, summaries["S1"].has_file, summaries["S1"].verifiable) == (8, False, True)
    assert (summaries["S2"].chars, summaries["S2"].has_file, summaries["S2"].verifiable) == (0, True, False)
    assert summaries["S2"].kind is SourceKind.IMAGE


def test_build_report_without_mapping() -> None:
    report = build_report(
        form_info=form_info(),
        form_name="form.pdf",
        results=[],
        sources=[],
        mapping=None,
        flattened=False,
        dry_run=True,
        warnings=[],
        timings_ms={},
    )
    assert report.model is None and report.prompt_version is None and report.usage == {}
    assert report.counts == {"filled": 0, "no_data": 0, "rejected": 0, "not_writable": 0}


def test_build_report_copies_results() -> None:
    original = results()
    report = build_report(
        form_info=form_info(),
        form_name="form.pdf",
        results=original,
        sources=[],
        mapping=None,
        flattened=False,
        dry_run=False,
        warnings=[],
        timings_ms={},
    )
    report.fields[0].value = "changed"
    assert original[0].value == SSN


def test_mask_report_masks_sensitive_values_and_hides_evidence() -> None:
    masked = mask_report(make_report())
    f1 = masked.fields[0]
    assert masked.masked
    assert f1.value == "***-**-6789"
    assert f1.evidence == HIDDEN_EVIDENCE == "[hidden: sensitive]"


def test_mask_report_scrubs_everything_else() -> None:
    masked = mask_report(make_report())
    f2, f3, f4 = masked.fields[1:]
    assert f2.value == "Jane Doe"
    assert f2.evidence == "Jane Doe, card **** **** **** 1111"
    assert f3.value == "******3210" and f3.evidence == "Account ******3210"
    assert f3.detail == "Confidence 0.40 is below 0.70"
    assert f4.value is None and f4.evidence is None
    assert masked.sources[1].name == "ssn ***-**-6789.txt"
    assert masked.form_name == "form for ***-**-6789.pdf"
    assert masked.warnings == ["Something about ****@******e.com and ***-**-6789"]


def test_mask_report_redacts_sensitive_values_across_other_fields() -> None:
    report = build_report(
        form_info=form_info(),
        form_name="form.pdf",
        results=[
            FieldResult(
                field_id="F1",
                name="dob",
                label="Date of birth",
                type=FieldType.TEXT,
                status=FieldStatus.FILLED,
                value="01/02/1980",
                source_id="S1",
                evidence="DOB: 01/02/1980",
                sensitive=True,
            ),
            FieldResult(
                field_id="F2",
                name="passport",
                label="Passport",
                type=FieldType.TEXT,
                status=FieldStatus.REJECTED,
                value="X1234567",
                source_id="S1",
                evidence="Passport X1234567",
                sensitive=True,
                reason=RejectReason.LOW_CONFIDENCE,
            ),
            FieldResult(
                field_id="F3",
                name="full_name",
                type=FieldType.TEXT,
                status=FieldStatus.FILLED,
                value="Jane Doe",
                source_id="S1",
                evidence="Applicant: Jane Doe, SSN 123.45.6789, DOB 01/02/1980, passport x 1234567",
                notes=["Also listed born January 2, 1980"],
            ),
            FieldResult(
                field_id="F4",
                name="date_signed",
                type=FieldType.TEXT,
                status=FieldStatus.FILLED,
                value="2026-10-06",
                source_id="S0",
                evidence="Today's date: 2026-10-06",
            ),
        ],
        sources=sources(),
        mapping=None,
        flattened=False,
        dry_run=False,
        warnings=[],
        timings_ms={},
    )
    masked = mask_report(report)
    by_name = {f.name: f for f in masked.fields}
    assert by_name["dob"].value == "**/**/****" and by_name["dob"].evidence == HIDDEN_EVIDENCE
    assert by_name["passport"].value == "****4567" and by_name["passport"].evidence == HIDDEN_EVIDENCE
    assert by_name["full_name"].evidence == "Applicant: Jane Doe, SSN ***.**.6789, DOB **/**/****, passport * ***4567"
    assert by_name["full_name"].notes == ["Also listed born ******* *, ****"]
    assert by_name["date_signed"].value == "2026-10-06"
    dumped = masked.model_dump_json()
    for secret in ("01/02/1980", "January 2, 1980", "123.45.6789", "X1234567", "x 1234567"):
        assert secret not in dumped


def test_masked_report_json_has_no_canaries() -> None:
    dumped = mask_report(make_report()).model_dump_json()
    for canary in (SSN, "123456789", CARD, "9876543210", "jane@example.com"):
        assert canary not in dumped


def test_mask_report_returns_a_copy() -> None:
    report = make_report()
    mask_report(report)
    assert report.fields[0].value == SSN
    assert report.fields[0].evidence == f"SSN: {SSN}"
    assert not report.masked


def test_mask_report_is_idempotent() -> None:
    once = mask_report(make_report())
    twice = mask_report(once)
    assert twice == once
    assert twice is not once
    assert twice.fields[0].value == "***-**-6789"  # not masked again into "*******6789"
