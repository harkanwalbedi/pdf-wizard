"""Draws each field's id ([F1], [F2], ...) as a red box on a copy of the blank form.

The copy is only sent to the model for schema enrichment, so it can match the
field catalog to what it sees on the page. Widget annotations are removed from
the copy so their (often opaque) appearances can't hide the boxes. Uses pypdf
only; no new dependency.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Sequence

from pypdf import PageObject, PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject,
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)

from ..models import FormField
from .limits import pypdf_limits

log = logging.getLogger(__name__)

_FONT = "/PWBadgeFont"
_BADGE_PT = 6.0


def page_sizes(form_pdf: bytes) -> dict[int, tuple[float, float]]:
    """1-based page number -> (width, height) in PDF points, from each page's MediaBox."""
    with pypdf_limits():
        reader = PdfReader(io.BytesIO(form_pdf), strict=False)
        return {
            i: (round(float(p.mediabox.width), 1), round(float(p.mediabox.height), 1))
            for i, p in enumerate(reader.pages, start=1)
        }


def render_field_overlay(form_pdf: bytes, fields: Sequence[FormField]) -> bytes:
    """Returns a copy of `form_pdf` with a labelled red box at each field's first widget.

    Raises on unreadable input; callers fall back to the plain PDF.
    """
    by_page: dict[int, list[FormField]] = {}
    for f in fields:
        if f.writable and f.page is not None and f.rect is not None:
            by_page.setdefault(f.page, []).append(f)

    with pypdf_limits():
        writer = PdfWriter(clone_from=PdfReader(io.BytesIO(form_pdf), strict=False))
        if "/AcroForm" in writer._root_object:  # the copy is an image of the form, not a form
            del writer._root_object["/AcroForm"]
        for page_no, page in enumerate(writer.pages, start=1):
            if "/Annots" in page:
                kept = ArrayObject(
                    a for a in page["/Annots"] if a.get_object().get("/Subtype") != "/Widget"
                )
                page[NameObject("/Annots")] = kept
            page_fields = by_page.get(page_no)
            if page_fields:
                page.merge_page(_badge_page(page, page_fields))
        buf = io.BytesIO()
        writer.write(buf)
        return buf.getvalue()


def _badge_page(page: PageObject, fields: Sequence[FormField]) -> PageObject:
    """A transparent page holding the boxes and id labels, in the same user space as `page`."""
    ops: list[str] = ["q", "0.5 w"]
    for f in fields:
        x0, y0, x1, y1 = f.rect  # type: ignore[misc]
        w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
        label = f.id  # F<number>: only [A-Za-z0-9], safe inside a PDF string
        text_w = _BADGE_PT * 0.6 * len(label) + 2
        ops += [
            "1 0 0 RG", f"{x0:.2f} {y0:.2f} {w:.2f} {h:.2f} re S",  # red outline
            "1 1 1 rg", f"{x0:.2f} {y1 - _BADGE_PT - 1:.2f} {text_w:.2f} {_BADGE_PT + 1:.2f} re f",  # white tab
            "1 0 0 rg", "BT", f"{_FONT} {_BADGE_PT:.1f} Tf",
            f"{x0 + 1:.2f} {y1 - _BADGE_PT:.2f} Td", f"({label}) Tj", "ET",
        ]
    ops.append("Q")

    badge = PageObject.create_blank_page(width=page.mediabox.width, height=page.mediabox.height)
    badge.mediabox = page.mediabox
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    badge[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject(_FONT): font}),
    })
    stream = DecodedStreamObject()
    stream.set_data("\n".join(ops).encode("ascii"))
    badge[NameObject("/Contents")] = stream
    return badge
