"""load_sources: user inputs -> the source bundle S0..Sn (design doc, stage 3)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date

from .. import APP_NAME
from ..config import Limits
from ..errors import InvalidInputError
from ..models import Source, SourceKind
from .common import (
    ROLES,
    Context,
    LoadedSources,
    SourceInput,
    clean_text,
    decode_utf8,
    display_name,
    format_bytes,
    invalid,
)
from .documents import load_document
from .pdfs import load_reference
from .structured import load_structured

log = logging.getLogger(__name__)

MAX_NOTES = 10
_WEEKDAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
_MONTHS = "January February March April May June July August September October November December".split()


def load_sources(inputs: Sequence[SourceInput], *, limits: Limits, today: date) -> LoadedSources:
    """Validates and normalizes the inputs.

    Returns S0 (today's date) followed by S1, S2, ... in input order. Empty
    notes are skipped and don't use an ID. Raises InvalidInputError (or
    UnsupportedFormError for a password-protected PDF) with a safe message.
    """
    _check_counts(inputs, limits)
    warnings: list[str] = []
    sources = [system_source(today)]
    note_number = 0
    for item in inputs:
        sid = f"S{len(sources)}"
        if item.role == "note":
            note_number += 1
            source = _load_note(item, sid, note_number, limits)
        else:
            source = _load_file(item, Context(sid, display_name(item.name), limits, warnings))
        if source is not None:
            sources.append(source)
            log.debug("loaded %s: kind=%s chars=%d file=%s", source.id, source.kind.value,
                      len(source.text or ""), source.data is not None)
    return LoadedSources(sources=sources, warnings=warnings)


def system_source(today: date) -> Source:
    """S0: facts supplied by PDF Wizard. Names are spelled out in English, independent of locale."""
    iso = f"{today.year:04d}-{today.month:02d}-{today.day:02d}"
    spelled = f"{_WEEKDAYS[today.weekday()]}, {_MONTHS[today.month - 1]} {today.day}, {today.year}"
    return Source(id="S0", kind=SourceKind.SYSTEM, name=APP_NAME, text=f"Today's date: {iso} ({spelled}).")


def _check_counts(inputs: Sequence[SourceInput], limits: Limits) -> None:
    for item in inputs:
        if item.role not in ROLES:
            raise ValueError(f"unknown source role: {item.role!r}")
    files = sum(1 for item in inputs if item.role != "note")
    if files > limits.max_source_files:
        raise InvalidInputError(f"Too many files (limit {limits.max_source_files}).")
    if len(inputs) - files > MAX_NOTES:
        raise InvalidInputError(f"Too many notes (limit {MAX_NOTES}).")
    if sum(len(item.content) for item in inputs) > limits.max_total_bytes:
        raise InvalidInputError(f"The files are too large in total (limit {format_bytes(limits.max_total_bytes)}).")


def _load_file(item: SourceInput, ctx: Context) -> Source:
    if len(item.content) > ctx.limits.max_file_bytes:
        raise invalid(ctx, f"is too large (limit {format_bytes(ctx.limits.max_file_bytes)})")
    if not item.content:
        raise invalid(ctx, "is empty")
    if item.role == "data":
        return load_structured(item.content, ctx, row=item.row)
    if item.role == "doc":
        return load_document(item.content, ctx)
    return load_reference(item.content, ctx)


def _load_note(item: SourceInput, sid: str, number: int, limits: Limits) -> Source | None:
    """Notes are named by their position among the notes: "Note 2" is the second note given."""
    name = f"Note {number}"
    too_long = InvalidInputError(f"{name} is too long (limit {limits.max_note_chars:,} characters).")
    if len(item.content) > limits.max_file_bytes:  # bounds the decoding work
        raise too_long
    text = decode_utf8(item.content)
    if text is None:
        raise InvalidInputError(f"{name} is not valid UTF-8 text.")
    text = clean_text(text).strip()
    if not text:
        return None
    if len(text) > limits.max_note_chars:
        raise too_long
    return Source(id=sid, kind=SourceKind.NOTE, name=name, text=text)
