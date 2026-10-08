"""Tests for GeminiMapper with a fake client (no network)."""

from __future__ import annotations

import json
import logging
import re
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from google.auth import exceptions as auth_exceptions
from google.genai import errors as genai_errors
from google.genai import types

from pdf_wizard.config import Settings
from pdf_wizard.errors import ConfigError, MappingError
from pdf_wizard.mapping import gemini
from pdf_wizard.mapping.base import MappingRequest
from pdf_wizard.mapping.gemini import (
    CREDENTIALS_NOT_FOUND,
    UNUSABLE_RESPONSE,
    GeminiMapper,
    UnusableResponseError,
    create_mapper,
)
from pdf_wizard.mapping.prompt import PROMPT_VERSION, SYSTEM_PROMPT
from pdf_wizard.models import FieldType, FormField, MappingResponse, Source, SourceKind

PDF_BYTES = b"%PDF-1.7 source"
FORM_BYTES = b"%PDF-1.7 form"
SECRET = "Jane Doe 123-45-6789"
PAYLOAD = {
    "fields": [
        {
            "field_id": "F1",
            "value": "Jane Doe",
            "source_id": "S1",
            "evidence": "Name: Jane Doe",
            "confidence": 0.97,
            "sensitive": False,
        }
    ]
}


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeModels:
    """Records generate_content calls; returns (or raises) the queued outcomes in order."""

    def __init__(self, *outcomes: Any):
        self.outcomes = list(outcomes)
        self.calls: list[SimpleNamespace] = []

    def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
        self.calls.append(SimpleNamespace(model=model, contents=contents, config=config))
        outcome = self.outcomes[min(len(self.calls), len(self.outcomes)) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, *outcomes: Any):
        self.models = FakeModels(*outcomes)


def response(
    payload: dict | str | None = None,
    *,
    parsed: Any = "auto",
    finish_reason: Any = types.FinishReason.STOP,
    usage: bool = True,
    block_reason: Any = None,
    with_candidate: bool = True,
) -> types.GenerateContentResponse:
    text = payload if isinstance(payload, str) else json.dumps(PAYLOAD if payload is None else payload)
    kwargs: dict[str, Any] = {}
    if with_candidate:
        content = types.Content(role="model", parts=[types.Part.from_text(text=text)] if text else [])
        kwargs["candidates"] = [types.Candidate(content=content, finish_reason=finish_reason)]
    if usage:
        kwargs["usage_metadata"] = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=1200, candidates_token_count=80, thoughts_token_count=40, total_token_count=1320
        )
    if block_reason is not None:
        kwargs["prompt_feedback"] = types.GenerateContentResponsePromptFeedback(block_reason=block_reason)
    result = types.GenerateContentResponse(**kwargs)
    if parsed == "auto":
        try:
            parsed = MappingResponse.model_validate_json(text)
        except ValueError:
            parsed = None
    result.parsed = parsed
    return result


def api_error(code: int) -> genai_errors.APIError:
    body = {"error": {"code": code, "message": f"Project secret-proj-42 rejected {SECRET}", "status": "X"}}
    return genai_errors.ClientError(code, body) if code < 500 else genai_errors.ServerError(code, body)


def request() -> MappingRequest:
    return MappingRequest(
        fields=[FormField(id="F1", name="name", label="Full name", type=FieldType.TEXT, page=1)],
        sources=[
            Source(id="S0", kind=SourceKind.SYSTEM, name="PDF Wizard", text="Today's date: 2026-10-06"),
            Source(id="S1", kind=SourceKind.TEXT, name="notes.txt", text=f"Name: {SECRET}"),
            Source(id="S2", kind=SourceKind.PDF, name="doc.pdf", text="", media_type="application/pdf", data=PDF_BYTES),
        ],
        form_pdf=FORM_BYTES,
    )


def settings(**kwargs: Any) -> Settings:
    return Settings(project="test-project", **kwargs)


def propose(client: FakeClient, **settings_kwargs: Any):
    return GeminiMapper(settings(**settings_kwargs), client=client).propose(request())


def nonce_of(call: SimpleNamespace) -> str:
    first = call.contents[0].parts[0].text
    match = re.search(r"carry the nonce \[([0-9a-f]+)\]", first)
    assert match
    return match.group(1)


# --------------------------------------------------------------------------- #
# Request
# --------------------------------------------------------------------------- #


def test_passes_schema_system_instruction_and_seed() -> None:
    client = FakeClient(response())
    propose(client)
    (call,) = client.models.calls
    config = call.config
    assert call.model == Settings().model
    assert config.response_schema is MappingResponse
    assert config.response_mime_type == "application/json"
    assert config.system_instruction == SYSTEM_PROMPT
    assert config.seed == 7
    assert config.automatic_function_calling.disable is True


def test_omits_temperature_when_none() -> None:
    client = FakeClient(response())
    propose(client, temperature=None)
    config = client.models.calls[0].config
    assert config.temperature is None
    assert "temperature" not in config.model_fields_set


def test_passes_temperature_when_set() -> None:
    client = FakeClient(response())
    propose(client, temperature=0.0)
    assert client.models.calls[0].config.temperature == 0.0


def test_omits_seed_when_none() -> None:
    client = FakeClient(response())
    propose(client, seed=None)
    assert "seed" not in client.models.calls[0].config.model_fields_set


def test_sends_one_user_turn_with_text_and_file_parts() -> None:
    client = FakeClient(response())
    propose(client)
    (content,) = client.models.calls[0].contents
    assert content.role == "user"
    texts = [p.text for p in content.parts if p.text is not None]
    files = [(p.inline_data.data, p.inline_data.mime_type) for p in content.parts if p.inline_data is not None]
    assert "FIELD CATALOG" in texts[0] and '"id":"F1"' in texts[0]
    assert files == [(PDF_BYTES, "application/pdf"), (FORM_BYTES, "application/pdf")]
    nonce = nonce_of(client.models.calls[0])
    assert len(nonce) == 16
    for source_id in ("S0", "S1", "S2"):
        assert any(f"BEGIN SOURCE {source_id} [{nonce}]" in t for t in texts)
    assert any(t == "TARGET FORM (visual context only, not a data source)" for t in texts)


def test_nonce_is_fresh_per_request() -> None:
    client = FakeClient(response())
    mapper = GeminiMapper(settings(), client=client)
    mapper.propose(request())
    mapper.propose(request())
    first, second = client.models.calls
    assert nonce_of(first) != nonce_of(second)


# --------------------------------------------------------------------------- #
# Response
# --------------------------------------------------------------------------- #


def test_returns_proposals_model_prompt_version_and_usage() -> None:
    result = propose(FakeClient(response()))
    assert [(p.field_id, p.value, p.source_id, p.confidence) for p in result.proposals] == [
        ("F1", "Jane Doe", "S1", 0.97)
    ]
    assert result.model == Settings().model
    assert result.prompt_version == PROMPT_VERSION
    assert result.usage == {"input_tokens": 1200, "output_tokens": 80, "thinking_tokens": 40, "total_tokens": 1320}


def test_parses_response_text_when_parsed_is_missing() -> None:
    result = propose(FakeClient(response(parsed=None)))
    assert [p.value for p in result.proposals] == ["Jane Doe"]


def test_accepts_a_parsed_dict() -> None:
    result = propose(FakeClient(response(parsed=PAYLOAD)))
    assert [p.value for p in result.proposals] == ["Jane Doe"]


def test_usage_is_empty_without_metadata() -> None:
    assert propose(FakeClient(response(usage=False))).usage == {}


def test_empty_proposal_list_is_fine() -> None:
    assert propose(FakeClient(response({"fields": []}))).proposals == []


@pytest.mark.parametrize(
    "fake_response",
    [
        pytest.param(response(f"not json {SECRET}"), id="invalid-json"),
        pytest.param(response({"fields": [{"field_id": "F1", "value": SECRET}]}), id="schema-mismatch"),
        pytest.param(response('{"fields": [', finish_reason=types.FinishReason.MAX_TOKENS), id="max-tokens"),
        pytest.param(response(finish_reason=types.FinishReason.MAX_TOKENS), id="max-tokens-even-if-parsed"),
        pytest.param(response(finish_reason=types.FinishReason.SAFETY), id="safety"),
        pytest.param(response(finish_reason="RECITATION"), id="recitation-as-string"),
        pytest.param(
            response(with_candidate=False, block_reason=types.BlockedReason.SAFETY), id="prompt-blocked"
        ),
        pytest.param(response(with_candidate=False), id="no-candidates"),
        pytest.param(response(""), id="empty-text"),
    ],
)
def test_unusable_responses_raise_mapping_error(fake_response: Any) -> None:
    with pytest.raises(MappingError) as caught:
        propose(FakeClient(fake_response))
    assert caught.value.user_message == UNUSABLE_RESPONSE
    cause = caught.value.__cause__
    assert isinstance(cause, UnusableResponseError)
    assert "Jane" not in str(cause) and "123-45-6789" not in str(cause)
    # Nothing that quotes the response (e.g. a pydantic ValidationError) is chained.
    assert cause.__cause__ is None
    assert cause.__context__ is None or cause.__suppress_context__


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("code", [400, 401, 403, 404, 429, 500, 503])
def test_api_errors_map_to_mapping_error_with_status_code_only(code: int) -> None:
    error = api_error(code)
    with pytest.raises(MappingError) as caught:
        propose(FakeClient(error))
    message = caught.value.user_message
    assert f"(HTTP {code})" in message
    assert "secret-proj" not in message and "Jane" not in message and "123-45-6789" not in message
    assert caught.value.__cause__ is error


def test_default_credentials_error_maps_to_config_error() -> None:
    error = auth_exceptions.DefaultCredentialsError("Could not automatically determine credentials.")
    with pytest.raises(ConfigError) as caught:
        propose(FakeClient(error))
    assert caught.value.user_message == CREDENTIALS_NOT_FOUND
    assert caught.value.user_message == (
        "Google Cloud credentials not found. Run 'gcloud auth application-default login'."
    )
    assert caught.value.__cause__ is error


def test_refresh_error_maps_to_config_error() -> None:
    with pytest.raises(ConfigError) as caught:
        propose(FakeClient(auth_exceptions.RefreshError("Reauthentication is needed.")))
    assert "gcloud auth application-default login" in caught.value.user_message


@pytest.mark.parametrize(
    "error, phrase",
    [
        (httpx.ReadTimeout("timed out"), "didn't respond in time"),
        (httpx.ConnectError("connection refused"), "Couldn't reach the AI service"),
        (auth_exceptions.TransportError("token endpoint unreachable"), "Couldn't reach the AI service"),
        (RuntimeError(f"unexpected {SECRET}"), "The request to the AI model failed"),
    ],
)
def test_network_and_unexpected_errors_map_to_mapping_error(error: Exception, phrase: str) -> None:
    with pytest.raises(MappingError) as caught:
        propose(FakeClient(error))
    assert phrase in caught.value.user_message
    assert "Jane" not in caught.value.user_message
    assert caught.value.__cause__ is error


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


def test_logs_hold_counts_but_no_content(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="pdf_wizard.mapping.gemini"):
        propose(FakeClient(response()))
    assert "input_tokens=1200" in caplog.text and "proposals=1" in caplog.text
    assert "Jane" not in caplog.text and "123-45-6789" not in caplog.text


def test_error_logs_hold_class_and_status_only(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="pdf_wizard.mapping.gemini"), pytest.raises(MappingError):
        propose(FakeClient(api_error(403)))
    assert "error=ClientError" in caplog.text and "status=403" in caplog.text
    assert "secret-proj" not in caplog.text and "Jane" not in caplog.text


def test_unusable_response_logs_hold_no_content(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="pdf_wizard.mapping.gemini"), pytest.raises(MappingError):
        propose(FakeClient(response(f"not json {SECRET}")))
    assert "problem=invalid JSON" in caplog.text
    assert "Jane" not in caplog.text and "123-45-6789" not in caplog.text


# --------------------------------------------------------------------------- #
# Client creation
# --------------------------------------------------------------------------- #


class ClientFactory:
    """Stands in for genai.Client: records constructor arguments, returns a FakeClient."""

    def __init__(self, error: Exception | None = None):
        self.calls: list[dict[str, Any]] = []
        self.error = error

    def __call__(self, **kwargs: Any) -> FakeClient:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return FakeClient(response())


@pytest.fixture
def factory(monkeypatch: pytest.MonkeyPatch) -> ClientFactory:
    fake = ClientFactory()
    monkeypatch.setattr(gemini.genai, "Client", fake)
    return fake


def test_vertex_client_is_created_lazily_once(factory: ClientFactory) -> None:
    mapper = GeminiMapper(settings(location="us-central1", request_timeout_s=120.0))
    assert factory.calls == []
    mapper.propose(request())
    mapper.propose(request())
    (kwargs,) = factory.calls
    assert kwargs["vertexai"] is True
    assert (kwargs["project"], kwargs["location"]) == ("test-project", "us-central1")
    assert "api_key" not in kwargs
    http_options = kwargs["http_options"]
    assert http_options.timeout == 120_000
    assert http_options.retry_options.attempts == gemini.RETRY_ATTEMPTS
    assert 429 in http_options.retry_options.http_status_codes
    assert 400 not in http_options.retry_options.http_status_codes


def test_gemini_api_backend_reads_key_at_client_creation(
    factory: ClientFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-value")
    mapper = GeminiMapper(Settings(backend="gemini-api"))
    assert factory.calls == []
    mapper.propose(request())
    (kwargs,) = factory.calls
    assert kwargs["api_key"] == "test-key-value"
    assert kwargs["vertexai"] is False
    assert "test-key-value" not in repr(vars(mapper))


def test_gemini_api_backend_without_key_is_config_error(
    factory: ClientFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        GeminiMapper(Settings(backend="gemini-api")).propose(request())
    assert factory.calls == []


@pytest.mark.parametrize(
    "error, expected",
    [
        (auth_exceptions.DefaultCredentialsError("no ADC"), CREDENTIALS_NOT_FOUND),
        (ValueError("Project and location or API key must be set"), "The AI backend settings are invalid"),
    ],
)
def test_client_creation_errors_map_to_config_error(
    monkeypatch: pytest.MonkeyPatch, error: Exception, expected: str
) -> None:
    monkeypatch.setattr(gemini.genai, "Client", ClientFactory(error))
    with pytest.raises(ConfigError) as caught:
        GeminiMapper(settings()).propose(request())
    assert caught.value.user_message.startswith(expected)
    assert caught.value.__cause__ is error


def test_create_mapper_validates_settings_first(factory: ClientFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigError):
        create_mapper(Settings(project=None))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        create_mapper(Settings(backend="gemini-api"))
    with pytest.raises(ConfigError):
        create_mapper(Settings(project="p", model=""))
    assert factory.calls == []


def test_create_mapper_returns_a_lazy_gemini_mapper(factory: ClientFactory) -> None:
    mapper = create_mapper(settings())
    assert isinstance(mapper, GeminiMapper)
    assert factory.calls == []
