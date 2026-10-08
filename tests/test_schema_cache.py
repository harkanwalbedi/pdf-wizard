"""Phase 2 tests: [F#] overlay, disk cache, per-form lock, page batching, auto trigger, CLI flags."""

from __future__ import annotations

import io
import os
import stat
import threading
import time
from pathlib import Path

from pypdf import PdfReader

from pdf_wizard.config import Settings
from pdf_wizard.mapping.fake import FakeMapper
from pdf_wizard.mapping.prompt import build_schema_prompt
from pdf_wizard.models import SchemaFieldEnrichment
from pdf_wizard.pdf.inspector import open_form
from pdf_wizard.pdf.overlay import page_sizes, render_field_overlay
from pdf_wizard.schema import (
    MAX_FIELDS_PER_ENRICHMENT,
    _page_batches,
    build_schema,
    detect_schema,
    needs_visual_enrichment,
    schema_cache,
)
from tests.pdf_factory import Spec, make_form


def _cryptic_form() -> bytes:
    return make_form([Spec(kind="text", name="f1_01[0]", y=700), Spec(kind="text", name="f1_02[0]", y=660)])


def _labeller(schema, pdf):
    return [SchemaFieldEnrichment(field_id=f.id, label=f"Label for {f.id}") for f in schema.fields]


# --------------------------------------------------------------------------- #
# overlay
# --------------------------------------------------------------------------- #


def test_overlay_draws_ids_and_removes_widgets() -> None:
    pdf = make_form([Spec(kind="text", name="name", tooltip="Name", y=700),
                     Spec(kind="checkbox", name="agree", tooltip="Agree", y=650)])
    form = open_form(pdf)
    out = render_field_overlay(form.plain_pdf(), form.info.fields)
    reader = PdfReader(io.BytesIO(out))
    assert "/AcroForm" not in reader.trailer["/Root"]
    page = reader.pages[0]
    annots = [a.get_object() for a in page.get("/Annots", [])]
    assert not any(a.get("/Subtype") == "/Widget" for a in annots)
    text = page.extract_text()
    assert "F1" in text and "F2" in text
    assert page_sizes(out) == page_sizes(pdf)


def test_schema_prompt_has_rects_page_sizes_and_box_explanation() -> None:
    pdf = make_form([Spec(kind="text", name="name", tooltip="Name", y=700)])
    form = open_form(pdf)
    parts = build_schema_prompt(form.info.fields, pdf)
    intro = parts[0].text or ""
    assert '"rect":[' in intro
    assert "page 1: 612 x 792" in intro
    assert "red box" in intro


# --------------------------------------------------------------------------- #
# disk cache
# --------------------------------------------------------------------------- #


def test_disk_cache_hit_skips_model_and_files_are_private(tmp_path: Path) -> None:
    pdf = _cryptic_form()
    settings = Settings(schema_cache_dir=str(tmp_path / "schemas"))
    schema_cache.clear()
    first = FakeMapper(schema_fn=_labeller)
    detect_schema(pdf, settings=settings, mapper=first, enrich=True)
    assert len(first.schema_requests) == 1

    files = list((tmp_path / "schemas").glob("*.json"))
    assert len(files) == 1
    assert stat.S_IMODE(files[0].stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "schemas").stat().st_mode) == 0o700

    schema_cache.clear()  # simulate a new process
    second = FakeMapper(schema_fn=_labeller)
    cached = detect_schema(pdf, settings=settings, mapper=second, enrich=True)
    assert second.schema_requests == []
    assert cached.enriched and cached.fields[0].label == "Label for F1"


def test_disk_cache_ignores_corrupt_file_and_other_models(tmp_path: Path) -> None:
    pdf = _cryptic_form()
    settings = Settings(schema_cache_dir=str(tmp_path))
    schema_cache.clear()
    detect_schema(pdf, settings=settings, mapper=FakeMapper(schema_fn=_labeller), enrich=True)
    (cache_file,) = tmp_path.glob("*.json")

    schema_cache.clear()
    other = FakeMapper(schema_fn=_labeller, model="other-model")
    detect_schema(pdf, settings=settings, mapper=other, enrich=True)
    assert len(other.schema_requests) == 1  # different model -> different cache entry

    cache_file.write_text("{not json")
    schema_cache.clear()
    again = FakeMapper(schema_fn=_labeller)
    schema = detect_schema(pdf, settings=settings, mapper=again, enrich=True)
    assert len(again.schema_requests) == 1 and schema.enriched


def test_settings_from_env_cache_dir(tmp_path: Path) -> None:
    assert Settings().schema_cache_dir is None
    assert Settings.from_env({"PDF_WIZARD_SCHEMA_CACHE": "0"}).schema_cache_dir is None
    assert Settings.from_env({"XDG_CACHE_HOME": str(tmp_path)}).schema_cache_dir == os.path.join(
        str(tmp_path), "pdf-wizard", "schemas"
    )
    assert Settings.from_env({"PDF_WIZARD_SCHEMA_CACHE_DIR": "/x/y"}).schema_cache_dir == "/x/y"


# --------------------------------------------------------------------------- #
# lock
# --------------------------------------------------------------------------- #


def test_concurrent_detection_of_same_form_calls_model_once() -> None:
    pdf = _cryptic_form()
    schema_cache.clear()

    def slow(schema, form_pdf):
        time.sleep(0.2)
        return _labeller(schema, form_pdf)

    mapper = FakeMapper(schema_fn=slow)
    results = []
    threads = [threading.Thread(target=lambda: results.append(detect_schema(pdf, mapper=mapper, enrich=True)))
               for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(mapper.schema_requests) == 1
    assert all(r.enriched for r in results)


# --------------------------------------------------------------------------- #
# batching and the auto trigger
# --------------------------------------------------------------------------- #


def test_large_forms_are_enriched_in_page_batches() -> None:
    per_page = MAX_FIELDS_PER_ENRICHMENT // 2 + 1  # two pages don't fit in one batch
    specs = [Spec(kind="text", name=f"p{p}_f{i}", tooltip=f"Field {p}-{i}", y=760 - (i % 70) * 10,
                  x=40 + (i // 70) * 280, width=200, height=8, page=p)
             for p in (1, 2) for i in range(per_page)]
    pdf = make_form(specs, pages=2)
    schema = build_schema(open_form(pdf).info)
    batches = _page_batches(schema.fields)
    assert [len(b) for b in batches] == [per_page, per_page]
    assert {f.page for f in batches[0]} == {1} and {f.page for f in batches[1]} == {2}

    schema_cache.clear()
    mapper = FakeMapper(schema_fn=_labeller)
    enriched = detect_schema(pdf, mapper=mapper, enrich=True)
    assert len(mapper.schema_requests) == 2
    assert all(f.label == f"Label for {f.id}" for f in enriched.fields if f.writable)


def test_numbered_duplicate_labels_trigger_auto_enrichment() -> None:
    pdf = make_form([Spec(kind="text", name="f1_07[0]", tooltip="Telephone No", y=700),
                     Spec(kind="text", name="f1_11[0]", tooltip="Telephone No", y=660)])
    schema = build_schema(open_form(pdf).info)
    assert any(f.label and f.label.endswith("(#2)") for f in schema.fields)
    assert needs_visual_enrichment(schema)

    clear = make_form([Spec(kind="text", name="name", tooltip="Full name", y=700)])
    assert not needs_visual_enrichment(build_schema(open_form(clear).info))


def test_cli_enrich_flags_are_mutually_exclusive() -> None:
    import pytest

    from pdf_wizard import cli

    parser = cli.build_parser()
    args = parser.parse_args(["fill", "f.pdf", "--no-enrich-schema", "--note", "x", "--dry-run"])
    assert args.no_enrich_schema and not args.enrich_schema
    with pytest.raises(SystemExit):
        parser.parse_args(["fill", "f.pdf", "--enrich-schema", "--no-enrich-schema"])
