"""Shared test fixtures."""

from __future__ import annotations

import pytest

from pdf_wizard.schema import schema_cache


@pytest.fixture(autouse=True)
def _isolate_schema_cache(monkeypatch: pytest.MonkeyPatch):
    """The in-memory schema cache is process-wide; clear it so tests can't leak schemas into each other.

    Also keep any Settings.from_env() in tests away from the real ~/.cache.
    """
    monkeypatch.setenv("PDF_WIZARD_SCHEMA_CACHE", "0")
    schema_cache.clear()
    yield
    schema_cache.clear()
