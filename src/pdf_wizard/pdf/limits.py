"""Caps on what pypdf may decode from untrusted PDFs (AUD-PARSE-003).

pypdf decodes streams lazily: the xref and object streams while reading, page
contents during text extraction, fonts and appearance streams while filling.
Its configuration lives in a ContextVar, so the caps must be active where the
decoding runs, not only where the reader was created. `open_form`,
`LoadedForm.plain_pdf`, `fill_pdf` and page text extraction each enter
`pypdf_limits()` themselves, in whichever thread they run.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pypdf
from pypdf import PageObject
from pypdf.errors import LimitReachedError
from pypdf.generic import ArrayObject

# Inflated bytes per stream. pypdf's default is 75 MB; form pages need far less.
MAX_STREAM_BYTES = 8_000_000
# Content streams of one page that text extraction reads.
MAX_CONTENT_STREAMS = 100

_CONFIGURATION: dict[str, Any] = {
    "zlib_maximum_output_length": MAX_STREAM_BYTES,
    "lzw_maximum_output_length": MAX_STREAM_BYTES,
    "run_length_maximum_output_length": MAX_STREAM_BYTES,
    "array_based_stream_maximum_output_length": MAX_STREAM_BYTES,
    # pypdf would run an external jbig2dec program for JBIG2 images. PDF Wizard
    # never needs image data, and parsing runs no subprocesses.
    "jbig2dec_binary": None,
}


@contextmanager
def pypdf_limits() -> Iterator[None]:
    """Applies the caps to pypdf work done in this block (this thread or task only)."""
    with pypdf.apply_configuration(**_CONFIGURATION):
        yield


def page_text(page: PageObject, **kwargs: Any) -> str:
    """`page.extract_text(**kwargs)` under the caps.

    Raises LimitReachedError for a page with more than MAX_CONTENT_STREAMS
    content streams, which callers handle like any other unreadable page.
    """
    with pypdf_limits():
        contents = page.get("/Contents")
        contents = contents.get_object() if contents is not None else None
        if isinstance(contents, ArrayObject) and len(contents) > MAX_CONTENT_STREAMS:
            raise LimitReachedError("Too many content streams on one page.")
        return page.extract_text(**kwargs) or ""
