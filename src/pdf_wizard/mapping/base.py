"""The Mapper interface. Production uses Gemini; tests use FakeMapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..models import FormField, MappingResult, Source


@dataclass(frozen=True)
class MappingRequest:
    fields: list[FormField]  # writable fields only: the catalog offered to the model
    sources: list[Source]  # S0 (system facts) first
    form_pdf: bytes | None = field(default=None, repr=False)  # target form, visual context only


class Mapper(Protocol):
    def propose(self, request: MappingRequest) -> MappingResult:
        """Returns proposals. Raises MappingError or ConfigError on failure."""
        ...
