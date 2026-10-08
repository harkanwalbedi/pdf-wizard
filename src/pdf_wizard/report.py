"""The fill report: per-field outcomes plus audit data (design doc, Section 4, stage 7).

`build_report` returns the full report; `mask_report` returns the copy that is
safe to display and store (sensitive values masked, evidence hidden, free text
scrubbed).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from . import __version__
from .models import FieldResult, FieldStatus, FieldType, FillReport, FormInfo, MappingResult, Source, SourceSummary
from .privacy import ValueRedactor, mask_value, scrub_text

HIDDEN_EVIDENCE = "[hidden: sensitive]"


def build_report(
    *,
    form_info: FormInfo,
    form_name: str,
    results: list[FieldResult],
    sources: list[Source],
    mapping: MappingResult | None,
    flattened: bool,
    dry_run: bool,
    warnings: list[str],
    timings_ms: dict[str, int],
) -> FillReport:
    """Builds the unmasked report. Pass it through `mask_report` before showing or storing it."""
    counts = {str(status): 0 for status in FieldStatus}
    for result in results:
        counts[str(result.status)] += 1
    return FillReport(
        tool_version=__version__,
        created_at=datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        form_name=form_name,
        form_sha256=form_info.sha256,
        page_count=form_info.page_count,
        model=mapping.model if mapping else None,
        prompt_version=mapping.prompt_version if mapping else None,
        flattened=flattened,
        dry_run=dry_run,
        masked=False,
        counts=counts,
        fields=[r.model_copy(deep=True) for r in results],
        sources=[
            SourceSummary(
                id=s.id,
                kind=s.kind,
                name=s.name,
                chars=len(s.text or ""),
                has_file=s.data is not None,
                verifiable=s.verifiable,
            )
            for s in sources
        ],
        warnings=list(warnings),
        timings_ms=dict(timings_ms),
        usage=dict(mapping.usage) if mapping else {},
    )


def mask_report(report: FillReport) -> FillReport:
    """Returns a masked deep copy (masked=True). Idempotent.

    Sensitive fields: value masked ("***-**-6789"), evidence hidden. Everything
    else that may hold personal data (values, evidence, notes, details, source
    and form names, warnings) has sensitive field values redacted first and
    SSN/card/account/birth-date/email shapes scrubbed.
    """
    masked = report.model_copy(deep=True)
    if report.masked:
        return masked
    redactor = ValueRedactor(
        f.value
        for f in masked.fields
        if f.sensitive and f.value and f.type not in (FieldType.CHECKBOX, FieldType.RADIO)
    )

    def clean(text: str) -> str:
        return scrub_text(redactor(text))

    masked.fields = [_mask_field(f, clean) for f in masked.fields]
    masked.sources = [s.model_copy(update={"name": clean(s.name)}) for s in masked.sources]
    masked.warnings = [clean(w) for w in masked.warnings]
    masked.form_name = clean(masked.form_name)
    masked.masked = True
    return masked


def _mask_field(result: FieldResult, clean: Callable[[str], str]) -> FieldResult:
    if result.sensitive:
        value = mask_value(result.value) if result.value else result.value
        evidence = HIDDEN_EVIDENCE if result.evidence else result.evidence
    else:
        value = clean(result.value) if result.value else result.value
        evidence = clean(result.evidence) if result.evidence else result.evidence
    # Details and notes never contain values by construction; scrub anyway.
    return result.model_copy(
        update={
            "value": value,
            "evidence": evidence,
            "detail": clean(result.detail) if result.detail else result.detail,
            "notes": [clean(n) for n in result.notes],
        }
    )
