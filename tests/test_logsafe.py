"""Tests for logsafe: the redaction filter and logging setup."""

from __future__ import annotations

import io
import logging
import sys
from collections.abc import Iterator

import pytest

from pdf_wizard.logsafe import LOG_LEVEL_ENV, RedactingFilter, configure_logging

CANARY = "123-45-6789"


@pytest.fixture
def capture() -> Iterator[tuple[logging.Logger, io.StringIO]]:
    """A logger whose only handler formats records (with tracebacks) into a buffer."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    handler.addFilter(RedactingFilter())
    logger = logging.getLogger("pdf_wizard.test_logsafe")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    yield logger, stream
    logger.removeHandler(handler)
    logger.propagate = True


@pytest.fixture
def clean_root() -> Iterator[logging.Logger]:
    """Gives the test a root logger without redacting handlers; restores all logging state after."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    third_party = {name: logging.getLogger(name).level for name in ("google_genai", "httpx")}
    # Earlier tests (e.g. the CLI's) may already have called configure_logging().
    for handler in handlers:
        if any(isinstance(f, RedactingFilter) for f in handler.filters):
            root.removeHandler(handler)
    yield root
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(level)
    for name, saved in third_party.items():
        logging.getLogger(name).setLevel(saved)


def test_filter_scrubs_message_and_arguments(capture) -> None:
    logger, stream = capture
    logger.warning("lookup failed for %s (card %s)", CANARY, "4111 1111 1111 1111")
    output = stream.getvalue()
    assert CANARY not in output and "4111 1111 1111 1111" not in output
    assert "***-**-6789" in output and "**** **** **** 1111" in output


def test_filter_scrubs_exception_text(capture) -> None:
    logger, stream = capture
    try:
        raise ValueError(f"bad value {CANARY}")
    except ValueError:
        logger.exception("parse failed")
    output = stream.getvalue()
    assert "ValueError" in output and "Traceback" in output
    assert CANARY not in output
    assert "***-**-6789" in output


def test_filter_formats_exception_now_and_clears_exc_info() -> None:
    try:
        raise RuntimeError(f"secret {CANARY}")
    except RuntimeError:
        exc_info = sys.exc_info()
    record = logging.LogRecord("x", logging.ERROR, __file__, 1, "failed %s", (CANARY,), exc_info)
    assert RedactingFilter().filter(record) is True
    assert record.args == ()
    assert record.msg == "failed ***-**-6789"
    assert record.exc_info is None
    assert record.exc_text and CANARY not in record.exc_text and "RuntimeError" in record.exc_text


def test_filter_never_drops_records_even_with_bad_arguments(capture) -> None:
    logger, stream = capture
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "two %s %s", ("one",), None)
    assert RedactingFilter().filter(record) is True
    assert record.args == ()
    logger.info("ok %s", "fine")
    assert "ok fine" in stream.getvalue()


def test_filter_works_with_caplog_when_attached_to_the_logger(caplog) -> None:
    logger = logging.getLogger("pdf_wizard.test_logsafe.caplog")
    redactor = RedactingFilter()
    logger.addFilter(redactor)
    try:
        with caplog.at_level(logging.INFO, logger=logger.name):
            logger.info("user ssn %s", CANARY)
    finally:
        logger.removeFilter(redactor)
    assert CANARY not in caplog.text
    assert "***-**-6789" in caplog.text


def redacting_handlers(root: logging.Logger) -> list[logging.Handler]:
    return [h for h in root.handlers if any(isinstance(f, RedactingFilter) for f in h.filters)]


def test_configure_logging_is_idempotent(clean_root, monkeypatch) -> None:
    monkeypatch.delenv(LOG_LEVEL_ENV, raising=False)
    configure_logging()
    configure_logging("INFO")
    (handler,) = redacting_handlers(clean_root)
    assert isinstance(handler, logging.StreamHandler)
    assert "%(message)s" in handler.formatter._fmt
    assert clean_root.level == logging.INFO


def test_configure_logging_reads_level_from_env(clean_root, monkeypatch) -> None:
    monkeypatch.setenv(LOG_LEVEL_ENV, "debug")
    configure_logging("WARNING")
    assert clean_root.level == logging.DEBUG
    # Third-party loggers that can log request details stay at WARNING or above.
    assert logging.getLogger("google_genai").level >= logging.WARNING
    assert logging.getLogger("httpx").level >= logging.WARNING


def test_configure_logging_ignores_invalid_env_level(clean_root, monkeypatch) -> None:
    monkeypatch.setenv(LOG_LEVEL_ENV, "chatty")
    configure_logging("ERROR")
    assert clean_root.level == logging.ERROR


def test_configured_handler_writes_scrubbed_text_to_current_stderr(clean_root, monkeypatch, capsys) -> None:
    monkeypatch.delenv(LOG_LEVEL_ENV, raising=False)
    configure_logging("WARNING")
    logging.getLogger("pdf_wizard.test_logsafe.root").warning("value %s", CANARY)
    err = capsys.readouterr().err
    assert "value ***-**-6789" in err
    assert CANARY not in err
