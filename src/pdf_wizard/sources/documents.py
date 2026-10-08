"""Documents (--doc): PDF, PNG, JPEG, email, or plain text.

The type comes from the content (magic bytes), never from the extension alone;
a ".eml" name is only a hint that a text file is an email.
"""

from __future__ import annotations

from ..errors import InvalidInputError
from ..models import Source, SourceKind
from .common import (
    JPEG_MAGIC,
    PNG_MAGIC,
    Context,
    decode_utf8,
    has_binary_bytes,
    has_disallowed_controls,
    invalid,
    is_pdf,
    normalize_newlines,
)
from .emails import load_email, looks_like_email
from .pdfs import load_pdf_document


def load_document(data: bytes, ctx: Context) -> Source:
    """Loads one document, detecting its type from its content."""
    if data.startswith(PNG_MAGIC):
        return _image(data, "image/png", ctx)
    if data.startswith(JPEG_MAGIC):
        return _image(data, "image/jpeg", ctx)
    if is_pdf(data):
        return load_pdf_document(data, ctx)
    # Emails may legitimately hold 8-bit text and escape sequences (ISO-2022-JP),
    # so only NUL bytes rule them out.
    if b"\x00" not in data and (ctx.name.lower().endswith(".eml") or looks_like_email(data)):
        return load_email(data, ctx)
    return _load_text(data, ctx)


def _image(data: bytes, media_type: str, ctx: Context) -> Source:
    # No text: code can't check evidence against images (Source.verifiable is False).
    return Source(id=ctx.sid, kind=SourceKind.IMAGE, name=ctx.name, data=data, media_type=media_type)


def _load_text(data: bytes, ctx: Context) -> Source:
    if has_binary_bytes(data):
        raise _unsupported(ctx)
    text = decode_utf8(data)
    if text is None:
        raise invalid(ctx, "is not UTF-8 text; save it as UTF-8 and try again")
    if has_disallowed_controls(text):  # C1 control characters
        raise _unsupported(ctx)
    text = normalize_newlines(text).strip()
    if not text:
        raise invalid(ctx, "has no text")
    return Source(id=ctx.sid, kind=SourceKind.TEXT, name=ctx.name, text=ctx.truncate(text))


def _unsupported(ctx: Context) -> InvalidInputError:
    return InvalidInputError(
        f"Unsupported document type for '{ctx.name}'. Use PDF, PNG, JPEG, plain text, or .eml files."
    )
