"""PDF documents (--doc) and reference forms (--reference).

A reference form is a previously filled PDF about the same person or entity,
so its field values may be copied. The model receives the PDF file itself plus
the field values as text (page text extraction doesn't include field values);
the page text layer is kept only to verify evidence.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from ..errors import PdfWizardError
from ..models import FieldType, FormField, Source, SourceKind
from ..pdf.inspector import LoadedForm, open_form
from ..pdf.limits import page_text, pypdf_limits
from .common import Context, clean_text, escape_newlines, invalid, is_pdf

log = logging.getLogger(__name__)

PDF_MEDIA_TYPE = "application/pdf"
REFERENCE_HEADER = "Field values from a previously filled form:"
DOCUMENT_FIELDS_HEADER = "Form field values in this PDF:"

_VALUE_TYPES = frozenset({FieldType.TEXT, FieldType.CHECKBOX, FieldType.RADIO, FieldType.CHOICE})
_OPTION_TYPES = frozenset({FieldType.CHECKBOX, FieldType.RADIO, FieldType.CHOICE})
_BUTTON_TYPES = frozenset({FieldType.CHECKBOX, FieldType.RADIO})


def load_pdf_document(data: bytes, ctx: Context) -> Source:
    """A PDF document: the file goes to the model; its text layer verifies evidence."""
    form = _open(data, ctx)
    layer = _text_layer(form, ctx)
    lines = field_value_lines(form.info.fields)
    text = ctx.truncate(DOCUMENT_FIELDS_HEADER + "\n" + "\n".join(lines)) if lines else None
    check_text = "\n\n".join(part for part in (layer, text) if part) or None
    return Source(
        id=ctx.sid,
        kind=SourceKind.PDF,
        name=ctx.name,
        text=text,
        check_text=check_text,
        data=_plain_pdf(form, ctx),
        media_type=PDF_MEDIA_TYPE,
    )


def load_reference(data: bytes, ctx: Context) -> Source:
    """A previously filled form: its field values as text, plus the file."""
    if not is_pdf(data):
        raise invalid(ctx, "is not a PDF; reference forms must be PDF files")
    form = _open(data, ctx)
    layer = _text_layer(form, ctx)
    lines = field_value_lines(form.info.fields)
    if lines:
        text: str | None = ctx.truncate(REFERENCE_HEADER + "\n" + "\n".join(lines))
        check_text = f"{text}\n\n{layer}" if layer else text
    else:  # flattened or blank: the page text (and the file) are all there is
        ctx.warn("no filled fields found; using the page text only.")
        text, check_text = None, layer or None
    return Source(
        id=ctx.sid,
        kind=SourceKind.REFERENCE_FORM,
        name=ctx.name,
        text=text,
        check_text=check_text,
        data=_plain_pdf(form, ctx),
        media_type=PDF_MEDIA_TYPE,
    )


def field_value_lines(fields: Iterable[FormField]) -> list[str]:
    """'"<label>" [<name>] = <value>' for every field that holds a value.

    Unchecked boxes ("Off"), signatures, buttons, and hidden, password, and
    file-select fields are skipped. For buttons and choice fields, the option's
    display label is appended: '= self (Self-employed)'.
    """
    lines: list[str] = []
    for f in fields:
        if f.type not in _VALUE_TYPES or f.hidden or f.password or f.file_select:
            continue
        value = clean_text(f.current_value or "").strip()
        if not value or (f.type in _BUTTON_TYPES and value.casefold() == "off"):
            continue
        shown = value
        if f.type in _OPTION_TYPES:
            label = next((o.label for o in f.options if o.value == f.current_value and o.label), None)
            label = clean_text(label or "").strip()
            if label and label.casefold() != value.casefold():
                shown = f"{value} ({label})"
        name = clean_text(f.name).strip()
        label_text = clean_text(f.label or "").strip() or name
        lines.append(f'"{escape_newlines(label_text)}" [{escape_newlines(name)}] = {escape_newlines(shown)}')
    return lines


def _open(data: bytes, ctx: Context) -> LoadedForm:
    try:
        return open_form(data, limits=ctx.limits, for_filling=False)
    except PdfWizardError as exc:
        # Same error type; name the file, since several PDFs may be uploaded.
        raise type(exc)(f"'{ctx.name}': {exc.user_message}") from exc
    except Exception as exc:
        log.info("PDF could not be opened: source=%s error=%s", ctx.sid, type(exc).__name__)
        raise invalid(ctx, "could not be read as a PDF") from exc


def _plain_pdf(form: LoadedForm, ctx: Context) -> bytes:
    try:
        return form.plain_pdf()
    except Exception as exc:  # re-writing a decrypted copy can fail on damaged files
        log.info("PDF could not be copied: source=%s error=%s", ctx.sid, type(exc).__name__)
        raise invalid(ctx, "could not be read as a PDF") from exc


def _text_layer(form: LoadedForm, ctx: Context) -> str:
    """Page text with "[Page N]" markers. Empty for scans, which keeps them unverifiable."""
    # TODO(security): text extraction has no time limit; a process sandbox with
    # rlimits is planned before hosted mode (design doc, Section 9).
    limit = ctx.limits.max_source_chars
    parts: list[str] = []
    size = 0
    failed = 0
    with pypdf_limits():  # page contents are decoded here, after open_form returned
        for number, page in enumerate(form.reader.pages, start=1):  # page count capped by open_form
            try:
                text = clean_text(page_text(page)).strip()
            except Exception as exc:  # a damaged or oversized content stream: skip the page
                failed += 1
                log.info("page text extraction failed: source=%s error=%s", ctx.sid, type(exc).__name__)
                continue
            if text:
                parts.append(f"[Page {number}]\n{text}")
                size += len(parts[-1]) + 2
                if size > limit:
                    break
    if failed:
        ctx.warn("text could not be extracted from some pages.")
    return ctx.truncate(
        "\n\n".join(parts), message=f"only the first {limit:,} characters of its text are used to check evidence."
    )
