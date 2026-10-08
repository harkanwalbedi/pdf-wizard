"""Tests for Stage 1 schema detection, enrichment, caching, and Stage 2 schema-aware guardrails."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from pdf_wizard import cli
from pdf_wizard.config import Settings
from pdf_wizard.errors import InvalidInputError
from pdf_wizard.guardrails import apply_guardrails
from pdf_wizard.mapping.fake import FakeMapper, proposal
from pdf_wizard.mapping.prompt import build_prompt, build_schema_prompt
from pdf_wizard.models import (
    FieldStatus,
    FieldType,
    FormField,
    FormSchema,
    RejectReason,
    SchemaFieldEnrichment,
    Source,
    SourceKind,
)
from pdf_wizard.pipeline import inspect_form
from pdf_wizard.schema import (
    _SchemaCache,
    apply_schema,
    build_schema,
    detect_schema,
    needs_visual_enrichment,
    parse_schema_json,
    schema_cache,
)
from tests.pdf_factory import Spec, make_form


def _make_gov_like_pdf() -> bytes:
    """Builds a synthetic IRS-style PDF with sections, mutex checkboxes, split SSN combs, and office-use fields."""
    return make_form([
        Spec(
            name="topmostSubform[0].Page1[0].f1_01[0]",
            kind="text",
            tooltip="Part I. Claimant Information. 1. Full legal name",
            y=700,
        ),
        Spec(
            name="topmostSubform[0].Page1[0].c1_1[0]",
            kind="checkbox",
            tooltip="3a. Individual/sole proprietor",
            y=660,
        ),
        Spec(
            name="topmostSubform[0].Page1[0].c1_1[1]",
            kind="checkbox",
            tooltip="3a. C corporation",
            y=640,
        ),
        Spec(
            name="topmostSubform[0].Page1[0].c1_1[2]",
            kind="checkbox",
            tooltip="3a. S corporation",
            y=620,
        ),
        Spec(
            name="topmostSubform[0].Page1[0].ssn_p1[0]",
            kind="text",
            tooltip="Social security number",
            max_len=3,
            flags="comb",
            y=580,
        ),
        Spec(
            name="topmostSubform[0].Page1[0].ssn_p2[0]",
            kind="text",
            tooltip="Social security number",
            max_len=2,
            flags="comb",
            y=560,
        ),
        Spec(
            name="topmostSubform[0].Page1[0].ssn_p3[0]",
            kind="text",
            tooltip="Social security number",
            max_len=4,
            flags="comb",
            y=540,
        ),
        Spec(
            name="topmostSubform[0].Page1[0].ein_p1[0]",
            kind="text",
            tooltip="Employer identification number",
            max_len=2,
            flags="comb",
            y=500,
        ),
        Spec(
            name="topmostSubform[0].Page1[0].ein_p2[0]",
            kind="text",
            tooltip="Employer identification number",
            max_len=7,
            flags="comb",
            y=480,
        ),
        Spec(
            name="topmostSubform[0].Page1[0].yes_box",
            kind="checkbox",
            tooltip="Are you subject to backup withholding? Yes",
            y=440,
        ),
        Spec(
            name="topmostSubform[0].Page1[0].no_box",
            kind="checkbox",
            tooltip="Are you subject to backup withholding? No",
            y=420,
        ),
        Spec(
            name="topmostSubform[0].Page1[0].agency_code",
            kind="text",
            tooltip="For office use only - examiner code",
            y=380,
        ),
    ])


def test_deterministic_schema_detection() -> None:
    pdf_bytes = _make_gov_like_pdf()
    schema_cache.clear()
    schema = detect_schema(pdf_bytes, enrich=False)

    assert not schema.enriched
    by_id = {f.id: f for f in schema.fields}

    # 1. Section inference from "Part I. Claimant Information." prefix
    assert by_id["F1"].section == "Part I. Claimant Information"
    assert by_id["F2"].section == "Part I. Claimant Information"

    # 2. Mutex checkbox groups: c1_1[0..2] -> G1, Yes/No pair -> G2
    assert schema.mutex_groups == {
        "G1": ["F2", "F3", "F4"],
        "G2": ["F10", "F11"],
    }
    assert by_id["F2"].group == "G1"
    assert by_id["F3"].group == "G1"
    assert by_id["F4"].group == "G1"
    assert by_id["F10"].group == "G2"
    assert by_id["F11"].group == "G2"

    # 3. Split comb disambiguation (SSN 3-2-4 and EIN 2-7)
    assert by_id["F5"].label == "Social security number (first 3 digits)"
    assert by_id["F6"].label == "Social security number (middle 2 digits)"
    assert by_id["F7"].label == "Social security number (last 4 digits)"
    assert by_id["F8"].label == "Employer identification number (first 2 digits)"
    assert by_id["F9"].label == "Employer identification number (last 7 digits)"

    # 4. Office-use-only detection
    assert by_id["F12"].office_use_only is True
    assert by_id["F1"].office_use_only is False


def test_needs_visual_enrichment_detects_cryptic_fields() -> None:
    clean_pdf = _make_gov_like_pdf()
    assert not needs_visual_enrichment(detect_schema(clean_pdf, use_cache=False))

    cryptic_pdf = make_form([
        Spec(name="f1_01[0]", kind="text", y=700),
        Spec(name="c1_01[0]", kind="checkbox", y=660),
    ])
    cryptic_schema = detect_schema(cryptic_pdf, use_cache=False)
    assert needs_visual_enrichment(cryptic_schema)


def test_visual_schema_enrichment_and_caching() -> None:
    cryptic_pdf = make_form([
        Spec(name="f1_01[0]", kind="text", y=700),
        Spec(name="cb_a", kind="checkbox", y=660),
        Spec(name="cb_b", kind="checkbox", y=640),
        Spec(name="agency_box", kind="text", y=600),
    ])
    schema_cache.clear()
    mapper = FakeMapper(
        schema_enrichments=[
            SchemaFieldEnrichment(
                field_id="F1",
                label="1. Applicant's date of birth",
                section="Part I: Applicant",
                format_hint="date:mm/dd/yyyy",
            ),
            SchemaFieldEnrichment(
                field_id="F2",
                label="2. Filing status: Single",
                section="Part I: Applicant",
                mutex_group="status_grp",
            ),
            SchemaFieldEnrichment(
                field_id="F3",
                label="2. Filing status: Married",
                section="Part I: Applicant",
                mutex_group="status_grp",
            ),
            SchemaFieldEnrichment(
                field_id="F4",
                label="For official use only",
                office_use_only=True,
            ),
        ]
    )

    enriched = detect_schema(cryptic_pdf, mapper=mapper, enrich=True)
    assert enriched.enriched is True
    assert enriched.model == "fake-model"
    assert len(mapper.schema_requests) == 1

    by_id = {f.id: f for f in enriched.fields}
    assert by_id["F1"].label == "1. Applicant's date of birth"
    assert by_id["F1"].section == "Part I: Applicant"
    assert by_id["F1"].format_hint == "date:mm/dd/yyyy"
    assert enriched.mutex_groups == {"G1": ["F2", "F3"]}
    assert by_id["F2"].group == "G1"
    assert by_id["F3"].group == "G1"
    assert by_id["F4"].office_use_only is True

    # Second call hits cache without calling the model again.
    cached = detect_schema(cryptic_pdf, mapper=mapper, enrich=True)
    assert len(mapper.schema_requests) == 1
    assert cached.enriched is True


def test_apply_schema_rejects_sha256_mismatch_and_preserves_physical_constraints() -> None:
    pdf_bytes = make_form([
        Spec(name="ssn", kind="text", tooltip="SSN", max_len=9, flags="comb", y=700),
        Spec(name="locked", kind="text", tooltip="Locked", flags="readOnly", y=660),
    ])
    info = inspect_form(pdf_bytes, settings=Settings())
    schema = build_schema(info)

    # 1. SHA-256 mismatch is rejected.
    wrong_sha = schema.model_copy(update={"form_sha256": "0" * 64})
    with pytest.raises(InvalidInputError, match="SHA-256 mismatch"):
        apply_schema(info, wrong_sha)

    # 2. Tampering with physical PDF constraints in schema JSON cannot override FormInfo.
    tampered_fields = [
        schema.fields[0].model_copy(
            update={
                "label": "Enriched SSN Label",
                "max_len": 999,
                "comb": False,
                "type": FieldType.CHECKBOX,
            }
        ),
        schema.fields[1].model_copy(
            update={
                "label": "Attempted Unlock",
                "read_only": False,
            }
        ),
    ]
    tampered_schema = schema.model_copy(update={"fields": tampered_fields})
    applied = apply_schema(info, tampered_schema)

    assert applied.fields[0].label == "Enriched SSN Label"
    assert applied.fields[0].max_len == 9
    assert applied.fields[0].comb is True
    assert applied.fields[0].type is FieldType.TEXT
    assert applied.fields[1].read_only is True
    assert applied.fields[1].writable is False


def test_guardrails_enforce_office_use_only_and_mutex_checkbox_groups() -> None:
    pdf_bytes = _make_gov_like_pdf()
    info = inspect_form(pdf_bytes, settings=Settings())
    sources = [
        Source(
            id="S1",
            name="intake.txt",
            kind=SourceKind.TEXT,
            text="Name: Jane Doe\nClassification: C corporation or S corporation\nCode: X99",
        )
    ]

    # Propose TWO checkboxes in mutex group G1 (F3 and F4) AND propose a value for office-use F12.
    proposals = [
        proposal("F1", "Jane Doe", evidence="Jane Doe"),
        proposal("F3", "checked", evidence="C corporation"),
        proposal("F4", "checked", evidence="S corporation"),
        proposal("F12", "X99", evidence="X99"),
    ]
    checked = apply_guardrails(
        info.fields, sources, proposals, min_confidence=0.7, min_confidence_unverifiable=0.9
    )

    assert checked.results["F1"].status is FieldStatus.FILLED
    # Both conflicting checkboxes in mutex group G1 are rejected!
    assert checked.results["F3"].status is FieldStatus.REJECTED
    assert checked.results["F3"].reason is RejectReason.INVALID_VALUE
    assert "Only one checkbox in this group" in (checked.results["F3"].detail or "")
    assert checked.results["F4"].status is FieldStatus.REJECTED
    assert checked.results["F4"].reason is RejectReason.INVALID_VALUE
    assert "F3" not in checked.values and "F4" not in checked.values

    # Office-use-only field F12 is rejected!
    assert checked.results["F12"].status is FieldStatus.REJECTED
    assert checked.results["F12"].reason is RejectReason.INVALID_VALUE
    assert "office use only" in (checked.results["F12"].detail or "").lower()
    assert "F12" not in checked.values

    # When only ONE checkbox in mutex group G1 is proposed, it is accepted.
    checked_single = apply_guardrails(
        info.fields,
        sources,
        [proposal("F3", "checked", evidence="C corporation")],
        min_confidence=0.7,
        min_confidence_unverifiable=0.9,
    )
    assert checked_single.results["F3"].status is FieldStatus.FILLED
    assert checked_single.results["F3"].value == "checked"


def test_prompt_catalog_includes_section_group_and_office_use() -> None:
    pdf_bytes = _make_gov_like_pdf()
    info = inspect_form(pdf_bytes, settings=Settings())
    prompt_parts = build_prompt(MappingRequest_for(info.fields), nonce="testnonce")
    catalog_text = "\n".join(p.text for p in prompt_parts if p.text)
    assert '"section":"Part I. Claimant Information"' in catalog_text
    assert '"group":"G1"' in catalog_text
    assert '"office_use_only":true' in catalog_text

    schema_parts = build_schema_prompt(info.fields, pdf_bytes)
    assert "FIELD CATALOG" in (schema_parts[0].text or "")


def MappingRequest_for(fields: list[FormField]):
    from pdf_wizard.mapping.base import MappingRequest

    return MappingRequest(
        fields=fields,
        sources=[Source(id="S1", name="note", kind=SourceKind.NOTE, text="hi")],
    )


def test_cli_schema_and_fill_with_schema(tmp_path: Path) -> None:
    pdf_bytes = _make_gov_like_pdf()
    form_path = tmp_path / "form.pdf"
    form_path.write_bytes(pdf_bytes)
    schema_path = tmp_path / "schema.json"
    out_pdf = tmp_path / "filled.pdf"
    data_path = tmp_path / "data.json"
    data_path.write_text(json.dumps({"name": "Jane Doe", "ssn": "123-45-6789"}), encoding="utf-8")

    # 1. Run `pdf-wizard schema form.pdf -o schema.json`
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    rc = cli.run(["schema", str(form_path), "-o", str(schema_path)], stdout=out_buf, stderr=err_buf)
    assert rc == 0
    assert schema_path.is_file()
    assert "2 mutex group(s)" in out_buf.getvalue()

    loaded_schema = parse_schema_json(schema_path.read_bytes())
    assert loaded_schema.form_sha256

    # 2. Run `pdf-wizard fill form.pdf --schema schema.json -o filled.pdf --data data.json`
    mapper = FakeMapper([
        proposal("F1", "Jane Doe", evidence="Jane Doe"),
        proposal("F5", "123", evidence="123-45-6789", sensitive=True),
        proposal("F6", "45", evidence="123-45-6789", sensitive=True),
        proposal("F7", "6789", evidence="123-45-6789", sensitive=True),
    ])
    out_buf2 = io.StringIO()
    rc2 = cli.run(
        ["fill", str(form_path), "--schema", str(schema_path), "-o", str(out_pdf), "--data", str(data_path)],
        mapper_factory=lambda _s: mapper,
        stdout=out_buf2,
        stderr=err_buf,
    )
    assert rc2 == 0
    assert out_pdf.is_file()
    # Because an explicit schema was passed, Stage 2 skipped uploading the visual form_pdf!
    assert len(mapper.requests) == 1
    assert mapper.requests[0].form_pdf is None


def test_schema_cache_lru_eviction() -> None:
    cache = _SchemaCache(max_entries=2)
    s1 = FormSchema(form_sha256="a" * 64, page_count=1, fields=[])
    s2 = FormSchema(form_sha256="b" * 64, page_count=1, fields=[])
    s3 = FormSchema(form_sha256="c" * 64, page_count=1, fields=[])
    cache.put(s1)
    cache.put(s2)
    assert cache.get("a" * 64) is not None  # refreshes s1
    cache.put(s3)  # evicts s2
    assert cache.get("b" * 64) is None
    assert cache.get("a" * 64) is not None
    assert cache.get("c" * 64) is not None
