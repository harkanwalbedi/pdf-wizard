"""Phase 1 tests: trust rules for AI schema enrichment and fallback when enrichment fails."""

from __future__ import annotations

import json

import pytest

from pdf_wizard.config import Settings
from pdf_wizard.errors import MappingError
from pdf_wizard.mapping.fake import FakeMapper, proposal
from pdf_wizard.models import FieldStatus, FormSchema, SchemaEnrichmentResponse, SchemaFieldEnrichment
from pdf_wizard.pipeline import FillOptions, fill_form
from pdf_wizard.schema import (
    ENRICHMENT_FAILED_WARNING,
    build_schema,
    detect_schema,
    merge_enrichment,
    schema_cache,
)
from pdf_wizard.pdf.inspector import open_form
from pdf_wizard.sources import SourceInput
from tests.pdf_factory import Spec, make_form


def _schema(specs: list[Spec], *, pages: int = 1) -> FormSchema:
    return build_schema(open_form(make_form(specs, pages=pages)).info)


def _merge(schema: FormSchema, *items: SchemaFieldEnrichment) -> FormSchema:
    return merge_enrichment(schema, SchemaEnrichmentResponse(fields=list(items)), model="m")


# --------------------------------------------------------------------------- #
# office_use_only
# --------------------------------------------------------------------------- #


def test_model_office_use_flag_is_accepted_without_regex_match() -> None:
    schema = _schema([Spec(kind="text", name="hdr_ein", tooltip="EIN", y=750), Spec(kind="text", name="name", y=700)])
    merged = _merge(schema, SchemaFieldEnrichment(field_id="F1", label="Page header EIN", office_use_only=True))
    by_id = {f.id: f for f in merged.fields}
    assert by_id["F1"].office_use_only is True
    assert by_id["F2"].office_use_only is False


def test_model_cannot_clear_deterministic_office_use_flag() -> None:
    schema = _schema([Spec(kind="text", name="code", tooltip="For office use only - code", y=700)])
    assert schema.fields[0].office_use_only is True
    merged = _merge(schema, SchemaFieldEnrichment(field_id="F1", label="Examiner code", office_use_only=False))
    assert merged.fields[0].office_use_only is True


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #


def test_field_id_only_and_blank_model_labels_are_ignored() -> None:
    schema = _schema([Spec(kind="text", name="a", tooltip="Full name", y=700), Spec(kind="text", name="b", tooltip="City", y=660)])
    merged = _merge(
        schema,
        SchemaFieldEnrichment(field_id="F1", label="F1"),
        SchemaFieldEnrichment(field_id="F2", label="   "),
    )
    assert [f.label for f in merged.fields] == ["Full name", "City"]


def test_colliding_model_labels_revert_and_stay_unique() -> None:
    schema = _schema([Spec(kind="text", name="d1_name", tooltip="Designee name", y=700),
                      Spec(kind="text", name="d2_name", tooltip="Second designee name", y=660)])
    merged = _merge(
        schema,
        SchemaFieldEnrichment(field_id="F1", label="Name"),
        SchemaFieldEnrichment(field_id="F2", label="Name"),
    )
    labels = [f.label for f in merged.fields]
    assert len(set(labels)) == 2
    assert labels == ["Designee name", "Second designee name"]


# --------------------------------------------------------------------------- #
# checkbox groups
# --------------------------------------------------------------------------- #


def test_model_group_named_like_deterministic_group_does_not_merge_into_it() -> None:
    schema = _schema([
        Spec(kind="checkbox", name="c1_1[0]", tooltip="Single", y=700),
        Spec(kind="checkbox", name="c1_1[1]", tooltip="Married", y=680),
        Spec(kind="checkbox", name="cb_x", tooltip="Option X", y=640),
        Spec(kind="checkbox", name="cb_y", tooltip="Option Y", y=620),
    ])
    assert schema.mutex_groups == {"G1": ["F1", "F2"]}
    merged = _merge(
        schema,
        SchemaFieldEnrichment(field_id="F3", mutex_group="G1"),
        SchemaFieldEnrichment(field_id="F4", mutex_group="G1"),
    )
    # Model "G1" is a separate group, not appended to the deterministic G1.
    assert merged.mutex_groups == {"G1": ["F1", "F2"], "G2": ["F3", "F4"]}


def test_model_group_across_pages_is_dropped_with_warning() -> None:
    schema = _schema(
        [Spec(kind="checkbox", name="p1_box", tooltip="A", y=700, page=1),
         Spec(kind="checkbox", name="p2_box", tooltip="B", y=700, page=2)],
        pages=2,
    )
    merged = _merge(
        schema,
        SchemaFieldEnrichment(field_id="F1", mutex_group="grp"),
        SchemaFieldEnrichment(field_id="F2", mutex_group="grp"),
    )
    assert merged.mutex_groups == {}
    assert any("different pages" in w for w in merged.warnings)


def test_model_group_with_text_field_is_dropped_with_warning() -> None:
    schema = _schema([Spec(kind="checkbox", name="box", tooltip="Other", y=700),
                      Spec(kind="text", name="other_text", tooltip="Specify", y=680)])
    merged = _merge(
        schema,
        SchemaFieldEnrichment(field_id="F1", mutex_group="grp"),
        SchemaFieldEnrichment(field_id="F2", mutex_group="grp"),
    )
    assert merged.mutex_groups == {}
    assert all(f.group is None for f in merged.fields)
    assert any("not checkboxes" in w for w in merged.warnings)


def test_model_group_overlapping_deterministic_group_is_dropped() -> None:
    schema = _schema([
        Spec(kind="checkbox", name="c1_1[0]", tooltip="Single", y=700),
        Spec(kind="checkbox", name="c1_1[1]", tooltip="Married", y=680),
        Spec(kind="checkbox", name="cb_z", tooltip="Other", y=640),
    ])
    merged = _merge(
        schema,
        SchemaFieldEnrichment(field_id="F2", mutex_group="g"),
        SchemaFieldEnrichment(field_id="F3", mutex_group="g"),
    )
    assert merged.mutex_groups == {"G1": ["F1", "F2"]}
    assert any("overlaps" in w for w in merged.warnings)


# --------------------------------------------------------------------------- #
# fallback when enrichment fails
# --------------------------------------------------------------------------- #


def _failing_mapper(**kwargs) -> FakeMapper:
    def boom(schema, pdf):
        raise MappingError("The AI model returned an unusable response.")

    return FakeMapper(schema_fn=boom, **kwargs)


def test_enrichment_failure_falls_back_to_basic_schema_and_is_not_cached() -> None:
    pdf = make_form([Spec(kind="text", name="f1_01[0]", y=700)])
    schema_cache.clear()
    mapper = _failing_mapper()
    schema = detect_schema(pdf, mapper=mapper, enrich=True)
    assert schema.enriched is False
    assert ENRICHMENT_FAILED_WARNING in schema.warnings

    # Not cached: the next call retries enrichment.
    detect_schema(pdf, mapper=mapper, enrich=True)
    assert len(mapper.schema_requests) == 2


def test_enrichment_failure_raises_in_strict_mode() -> None:
    pdf = make_form([Spec(kind="text", name="f1_01[0]", y=700)])
    schema_cache.clear()
    with pytest.raises(MappingError):
        detect_schema(pdf, mapper=_failing_mapper(), enrich=True, strict=True)


def test_fill_still_works_when_enrichment_fails() -> None:
    pdf = make_form([Spec(kind="text", name="full_name", tooltip="Full name", y=700)])
    schema_cache.clear()
    mapper = _failing_mapper(proposals=[proposal("F1", "Ada Lovelace", evidence="applicant.name")])
    outcome = fill_form(
        pdf,
        [SourceInput("data", "a.json", json.dumps({"applicant": {"name": "Ada Lovelace"}}).encode())],
        mapper=mapper,
        settings=Settings(),
        options=FillOptions(enrich_schema=True),
    )
    assert outcome.report.fields[0].status is FieldStatus.FILLED
    assert ENRICHMENT_FAILED_WARNING in outcome.report.warnings
    # Fallback schema is not enriched, so the fill call still gets the blank form for visual context.
    assert mapper.requests[0].form_pdf is not None
