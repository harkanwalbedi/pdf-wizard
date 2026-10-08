"""A deterministic Mapper for tests (no network)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ..models import (
    FieldProposal,
    FormSchema,
    MappingResult,
    SchemaEnrichmentResponse,
    SchemaFieldEnrichment,
)
from .base import MappingRequest


class FakeMapper:
    """Returns canned proposals, or proposals computed by `fn(request)`.

    Records every request in `self.requests` so tests can inspect what the
    model would have been sent. Also supports `enrich_schema` for Stage 1
    schema-enrichment tests.
    """

    def __init__(
        self,
        proposals: Sequence[FieldProposal] = (),
        *,
        fn: Callable[[MappingRequest], list[FieldProposal]] | None = None,
        schema_enrichments: Sequence[SchemaFieldEnrichment] = (),
        schema_fn: Callable[[FormSchema, bytes], list[SchemaFieldEnrichment]] | None = None,
        error: Exception | None = None,
        model: str = "fake-model",
    ):
        self._proposals = list(proposals)
        self._fn = fn
        self._schema_enrichments = list(schema_enrichments)
        self._schema_fn = schema_fn
        self._error = error
        self.model = model
        self.requests: list[MappingRequest] = []
        self.schema_requests: list[tuple[FormSchema, bytes]] = []

    def propose(self, request: MappingRequest) -> MappingResult:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        proposals = self._fn(request) if self._fn else list(self._proposals)
        return MappingResult(proposals=proposals, model=self.model, prompt_version="fake")

    def enrich_schema(
        self, schema: FormSchema, form_pdf: bytes
    ) -> tuple[SchemaEnrichmentResponse, str]:
        self.schema_requests.append((schema, form_pdf))
        if self._error is not None:
            raise self._error
        items = (
            self._schema_fn(schema, form_pdf)
            if self._schema_fn
            else list(self._schema_enrichments)
        )
        return SchemaEnrichmentResponse(fields=items), self.model


def proposal(
    field_id: str,
    value: str,
    *,
    source_id: str = "S1",
    evidence: str = "",
    confidence: float = 0.95,
    sensitive: bool = False,
) -> FieldProposal:
    """Shorthand for building a FieldProposal in tests."""
    return FieldProposal(
        field_id=field_id,
        value=value,
        source_id=source_id,
        evidence=evidence or value,
        confidence=confidence,
        sensitive=sensitive,
    )
