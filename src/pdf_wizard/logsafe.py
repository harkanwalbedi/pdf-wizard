"""Logging setup with a redaction filter as a privacy backstop.

Code must not log personal data in the first place (log IDs, counts, timings,
and exception class names only). `RedactingFilter` is the safety net for
mistakes: it masks SSN, card, account-number, and email shapes in messages and
exception text before any handler formats them.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TextIO

from .privacy import scrub_text

LOG_LEVEL_ENV = "PDF_WIZARD_LOG_LEVEL"
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
# Third-party loggers whose debug output can include request details. They are
# never set below WARNING by configure_logging().
_THIRD_PARTY_LOGGERS = ("google_genai", "google.auth", "httpx", "httpcore", "urllib3")


class RedactingFilter(logging.Filter):
    """Scrubs the formatted message and any exception text of a log record. Never drops records."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # mismatched format arguments: keep the template only
            message = str(record.msg)
        record.msg = scrub_text(message)
        record.args = ()
        if record.exc_info and record.exc_info[0] is not None:
            # Format the traceback now, so handlers only ever see the scrubbed text.
            record.exc_text = scrub_text(logging.Formatter().formatException(record.exc_info))
            record.exc_info = None
        elif record.exc_text:
            record.exc_text = scrub_text(record.exc_text)
        if record.stack_info:
            record.stack_info = scrub_text(record.stack_info)
        return True


class _RedactingHandler(logging.StreamHandler):
    """Writes to the current sys.stderr, looked up per record (like logging.lastResort).

    Also a marker type, so configure_logging() can find the handler it installed.
    """

    def __init__(self) -> None:
        logging.Handler.__init__(self)

    @property
    def stream(self) -> TextIO:  # type: ignore[override]
        return sys.stderr


def configure_logging(level: str | int = "WARNING") -> None:
    """Installs one stderr handler with RedactingFilter on the root logger. Idempotent.

    PDF_WIZARD_LOG_LEVEL, if set to a valid level, overrides `level`.
    """
    resolved = _resolve_level(os.environ.get(LOG_LEVEL_ENV)) or _resolve_level(level) or logging.WARNING
    root = logging.getLogger()
    handler = next((h for h in root.handlers if isinstance(h, _RedactingHandler)), None)
    if handler is None:
        handler = _RedactingHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        handler.addFilter(RedactingFilter())
        root.addHandler(handler)
    root.setLevel(resolved)
    for name in _THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(max(resolved, logging.WARNING))


def _resolve_level(value: str | int | None) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = value.strip().upper()
    if text.isdigit():
        return int(text)
    return logging.getLevelNamesMapping().get(text) if text else None
