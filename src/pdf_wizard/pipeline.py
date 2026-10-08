"""The fill pipeline (design doc, Section 4).

validate -> inspect -> normalize sources -> map (one model call) -> guardrails
-> fill -> report. The CLI and the web app both call `fill_form`, so they
behave the same way.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from .config import MB, Settings
from .errors import InvalidInputError, UnsupportedFormError
from .guardrails import apply_guardrails
from .mapping.base import Mapper, MappingRequest
from .models import FieldStatus, FillReport, FormInfo, FormSchema, RejectReason, SourceKind
from .pdf.filler import fill_pdf
from .pdf.inspector import open_form
from .report import build_report, mask_report
from .schema import apply_schema, build_schema, detect_schema
from .sources import SourceInput, load_sources

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FillOptions:
    flatten: bool = False
    dry_run: bool = False  # map and check values, but don't produce a PDF
    min_confidence: float | None = None  # overrides Settings.min_confidence
    schema: FormSchema | None = None  # pre-detected FormSchema (skips Stage 1 and redundant visual PDF upload)
    enrich_schema: bool | Literal["auto"] = False  # run visual LLM Stage 1 schema enrichment before filling


@dataclass
class FillOutcome:
    pdf: bytes | None = field(repr=False)  # None for dry runs; holds every value in clear text
    report: FillReport  # masked: safe to display and store
    # Field name -> display value of every filled field, unmasked. For the eval
    # harness only; never log or display it.
    raw_values: dict[str, str] = field(default_factory=dict, repr=False)


def inspect_form(form_bytes: bytes, *, settings: Settings) -> FormInfo:
    info = open_form(form_bytes, limits=settings.limits).info
    return apply_schema(info, build_schema(info))


def detect_form_schema(
    form_bytes: bytes,
    *,
    settings: Settings,
    mapper: Mapper | None = None,
    enrich: bool | Literal["auto"] = False,
) -> FormSchema:
    return detect_schema(form_bytes, settings=settings, mapper=mapper, enrich=enrich)


def fill_form(
    form_bytes: bytes,
    inputs: Sequence[SourceInput],
    *,
    mapper: Mapper,
    settings: Settings,
    options: FillOptions = FillOptions(),
    form_name: str = "form.pdf",
    today: date | None = None,
) -> FillOutcome:
    limits = settings.limits
    timings: dict[str, int] = {}
    started = last = time.perf_counter()

    def lap(stage: str) -> None:
        nonlocal last
        now = time.perf_counter()
        timings[stage] = round((now - last) * 1000)
        last = now

    total_bytes = len(form_bytes) + sum(len(i.content) for i in inputs)
    if total_bytes > limits.max_total_bytes:
        raise InvalidInputError(
            f"The files are too large in total (limit {limits.max_total_bytes // MB} MB)."
        )

    # 1-2. Validate, inspect, and apply/detect the target form's schema.
    form = open_form(form_bytes, limits=limits)
    if options.schema is not None:
        schema = options.schema
        form.info = apply_schema(form.info, schema)
    else:
        schema = detect_schema(form, settings=settings, mapper=mapper, enrich=options.enrich_schema)
        form.info = apply_schema(form.info, schema)
    catalog = [f for f in form.info.fields if f.writable]
    if not catalog:
        raise UnsupportedFormError("This PDF has no fields that can be filled.")
    lap("inspect")

    # 3. Normalize sources.
    loaded = load_sources(inputs, limits=limits, today=today or date.today())
    if not any(s.kind is not SourceKind.SYSTEM for s in loaded.sources):
        raise InvalidInputError("Add at least one data file, document, reference form, or note.")
    lap("sources")

    # 4. Map: one model call against the schema catalog.
    # When an explicit or visually-enriched schema is already present, skip
    # re-uploading the visual PDF bytes during the fill call.
    include_visual = settings.include_form_visual and options.schema is None and not schema.enriched
    request = MappingRequest(
        fields=catalog,
        sources=loaded.sources,
        form_pdf=form.plain_pdf() if include_visual else None,
    )
    mapping = mapper.propose(request)
    lap("mapping")

    # 5. Guardrails.
    min_conf = settings.min_confidence if options.min_confidence is None else options.min_confidence
    checked = apply_guardrails(
        form.info.fields,
        loaded.sources,
        mapping.proposals,
        min_confidence=min_conf,
        min_confidence_unverifiable=max(settings.min_confidence_unverifiable, min_conf),
    )
    results = dict(checked.results)
    values = dict(checked.values)
    warnings = [*form.info.warnings, *loaded.warnings, *checked.warnings]
    lap("guardrails")

    # 6. Fill.
    pdf: bytes | None = None
    if not options.dry_run:
        filled = fill_pdf(form, values, flatten=options.flatten)
        for field_id, detail in filled.failed.items():
            values.pop(field_id, None)
            results[field_id] = results[field_id].model_copy(
                update={"status": FieldStatus.REJECTED, "reason": RejectReason.RENDER_FAILED, "detail": detail}
            )
        warnings.extend(filled.warnings)
        pdf = filled.pdf
        lap("fill")

    # 7. Report.
    timings["total"] = round((time.perf_counter() - started) * 1000)
    ordered = [results[f.id] for f in form.info.fields]
    report = build_report(
        form_info=form.info,
        form_name=form_name,
        results=ordered,
        sources=loaded.sources,
        mapping=mapping,
        flattened=options.flatten and not options.dry_run,
        dry_run=options.dry_run,
        warnings=warnings,
        timings_ms=timings,
    )
    raw_values = {r.name: r.value for r in ordered if r.status is FieldStatus.FILLED and r.value is not None}
    log.info(
        "fill finished: fields=%d filled=%d rejected=%d no_data=%d total_ms=%d",
        len(ordered),
        report.counts.get(FieldStatus.FILLED, 0),
        report.counts.get(FieldStatus.REJECTED, 0),
        report.counts.get(FieldStatus.NO_DATA, 0),
        timings["total"],
    )
    return FillOutcome(pdf=pdf, report=mask_report(report), raw_values=raw_values)
