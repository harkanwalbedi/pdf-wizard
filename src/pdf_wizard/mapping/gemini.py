"""GeminiMapper: one structured-output Gemini call per fill (design doc, Section 5).

Logs hold only the model ID, timings, token counts, finish reasons, and error
class names and HTTP status codes. Prompts, responses, and SDK error text
(which can echo request details) are never logged or shown to the user.
"""

from __future__ import annotations

import logging
import os
import secrets
import threading
import time
from typing import Any

import httpx
from google import genai
from google.auth import exceptions as auth_exceptions
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import ValidationError

from ..config import Settings
from ..errors import ConfigError, MappingError, PdfWizardError
from ..models import FormSchema, MappingResponse, MappingResult, SchemaEnrichmentResponse
from ..schema import SCHEMA_SYSTEM_PROMPT
from .base import MappingRequest
from .prompt import PROMPT_VERSION, SYSTEM_PROMPT, PromptPart, build_prompt, build_schema_prompt

log = logging.getLogger(__name__)

UNUSABLE_RESPONSE = "The AI model returned an unusable response. Try again, or try fewer or smaller documents."
CREDENTIALS_NOT_FOUND = "Google Cloud credentials not found. Run 'gcloud auth application-default login'."
CREDENTIALS_INVALID = (
    "Google Cloud credentials have expired or are invalid. Run 'gcloud auth application-default login'."
)

# The SDK retries rate limits, server errors, and network timeouts with
# exponential backoff. Settings.request_timeout_s applies to each attempt.
RETRY_ATTEMPTS = 3  # including the first request
RETRY_STATUS_CODES = (408, 429, 500, 502, 503, 504)
_RETRY_INITIAL_DELAY_S = 2.0
_RETRY_MAX_DELAY_S = 16.0

_MAX_RESPONSE_CHARS = 5_000_000  # larger response text is not parsed
_USAGE_FIELDS = {
    "input_tokens": "prompt_token_count",
    "output_tokens": "candidates_token_count",
    "thinking_tokens": "thoughts_token_count",
    "total_tokens": "total_token_count",
}


class UnusableResponseError(Exception):
    """Diagnostic cause chained to MappingError. Holds no response content."""


class GeminiMapper:
    """Mapper backed by Gemini: Vertex AI with ADC by default, or the Gemini API.

    `client` is injectable for tests; otherwise a google-genai client is
    created on the first `propose()` call.
    """

    def __init__(self, settings: Settings, *, client: Any | None = None):
        self._settings = settings
        self._client = client
        self._client_lock = threading.Lock()

    def propose(self, request: MappingRequest) -> MappingResult:
        """Sends the catalog and sources in one request and returns the model's proposals.

        Raises MappingError (API failure, unusable response) or ConfigError
        (missing or invalid credentials or settings).
        """
        client = self._get_client()
        parts = build_prompt(request, nonce=secrets.token_hex(8))
        contents = [types.Content(role="user", parts=[_sdk_part(p) for p in parts])]
        model = self._settings.model
        started = time.perf_counter()
        try:
            response = client.models.generate_content(model=model, contents=contents, config=self._config())
        except Exception as exc:
            log.warning(
                "gemini call failed: model=%s ms=%d error=%s status=%s",
                model,
                _elapsed_ms(started),
                type(exc).__name__,
                _status_code(exc),
            )
            raise _translate_error(exc) from exc
        elapsed = _elapsed_ms(started)
        usage = _usage(response)
        try:
            parsed = _parse_response(response)
        except UnusableResponseError as exc:
            log.warning("gemini response unusable: model=%s ms=%d problem=%s", model, elapsed, exc)
            raise MappingError(UNUSABLE_RESPONSE) from exc
        log.info(
            "gemini call finished: model=%s ms=%d proposals=%d input_tokens=%s output_tokens=%s "
            "thinking_tokens=%s total_tokens=%s",
            model,
            elapsed,
            len(parsed.fields),
            *(usage.get(key, "-") for key in _USAGE_FIELDS),
        )
        return MappingResult(proposals=list(parsed.fields), model=model, prompt_version=PROMPT_VERSION, usage=usage)

    def enrich_schema(self, schema: FormSchema, form_pdf: bytes) -> tuple[SchemaEnrichmentResponse, str]:
        """Visually analyzes a blank PDF form to enrich field labels, sections, and mutex groups."""
        client = self._get_client()
        parts = build_schema_prompt(schema.fields, form_pdf)
        contents = [types.Content(role="user", parts=[_sdk_part(p) for p in parts])]
        model = self._settings.model
        started = time.perf_counter()
        config = self._config(system_prompt=SCHEMA_SYSTEM_PROMPT, response_schema=SchemaEnrichmentResponse)
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as exc:
            log.warning(
                "gemini schema call failed: model=%s ms=%d error=%s status=%s",
                model,
                _elapsed_ms(started),
                type(exc).__name__,
                _status_code(exc),
            )
            raise _translate_error(exc) from exc
        elapsed = _elapsed_ms(started)
        try:
            parsed = _parse_response(response, schema_cls=SchemaEnrichmentResponse)
        except UnusableResponseError as exc:
            log.warning("gemini schema response unusable: model=%s ms=%d problem=%s", model, elapsed, exc)
            raise MappingError(UNUSABLE_RESPONSE) from exc
        log.info("gemini schema call finished: model=%s ms=%d fields=%d", model, elapsed, len(parsed.fields))
        return parsed, model

    def generate(
        self, parts: list[PromptPart], *, system_prompt: str, response_schema: Any, model: str | None = None
    ) -> Any:
        """One structured call with a caller-supplied prompt and schema (used by eval tooling)."""
        client = self._get_client()
        contents = [types.Content(role="user", parts=[_sdk_part(p) for p in parts])]
        model = model or self._settings.model
        started = time.perf_counter()
        config = self._config(system_prompt=system_prompt, response_schema=response_schema)
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as exc:
            log.warning("gemini generate failed: model=%s error=%s", model, type(exc).__name__)
            raise _translate_error(exc) from exc
        try:
            return _parse_response(response, schema_cls=response_schema)
        except UnusableResponseError as exc:
            log.warning("gemini generate response unusable: model=%s ms=%d problem=%s", model, _elapsed_ms(started), exc)
            raise MappingError(UNUSABLE_RESPONSE) from exc

    def _config(
        self,
        *,
        system_prompt: str = SYSTEM_PROMPT,
        response_schema: Any = MappingResponse,
    ) -> types.GenerateContentConfig:
        options: dict[str, Any] = {
            "system_instruction": system_prompt,
            "response_mime_type": "application/json",
            "response_schema": response_schema,
            # No tools are declared; this also skips the SDK's function-calling loop.
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        # None means "use the model's default", so the parameter is left out.
        if self._settings.temperature is not None:
            options["temperature"] = self._settings.temperature
        if self._settings.seed is not None:
            options["seed"] = self._settings.seed
        return types.GenerateContentConfig(**options)

    def _get_client(self) -> Any:
        with self._client_lock:
            if self._client is None:
                self._client = _create_client(self._settings)
            return self._client


def create_mapper(settings: Settings) -> GeminiMapper:
    """Returns a GeminiMapper after checking that the backend is configured (fails closed)."""
    settings.validate_for_model_calls()
    return GeminiMapper(settings)


def _create_client(settings: Settings) -> Any:
    http_options = types.HttpOptions(
        timeout=round(settings.request_timeout_s * 1000),
        retry_options=types.HttpRetryOptions(
            attempts=RETRY_ATTEMPTS,
            initial_delay=_RETRY_INITIAL_DELAY_S,
            max_delay=_RETRY_MAX_DELAY_S,
            http_status_codes=list(RETRY_STATUS_CODES),
        ),
    )
    try:
        if settings.backend == "gemini-api":
            # Read only here and handed to the SDK client; PDF Wizard never stores or logs it.
            api_key = os.environ.get("GEMINI_API_KEY", "").strip()
            if not api_key:
                raise ConfigError("Set GEMINI_API_KEY (paid tier only), or use the default Vertex AI backend.")
            # vertexai=False explicitly, so environment variables can't switch the backend.
            return genai.Client(vertexai=False, api_key=api_key, http_options=http_options)
        return genai.Client(
            vertexai=True, project=settings.project, location=settings.location, http_options=http_options
        )
    except PdfWizardError:
        raise
    except auth_exceptions.DefaultCredentialsError as exc:
        raise ConfigError(CREDENTIALS_NOT_FOUND) from exc
    except ValueError as exc:
        log.warning("gemini client creation failed: error=%s", type(exc).__name__)
        raise ConfigError(
            "The AI backend settings are invalid. Check PDF_WIZARD_BACKEND, GOOGLE_CLOUD_PROJECT, "
            "and GOOGLE_CLOUD_LOCATION."
        ) from exc


def _sdk_part(part: PromptPart) -> types.Part:
    if part.data is not None:
        return types.Part.from_bytes(data=part.data, mime_type=part.mime_type or "application/octet-stream")
    return types.Part.from_text(text=part.text or "")


def _parse_response(response: Any, *, schema_cls: Any = MappingResponse) -> Any:
    """Returns the validated response. Raises UnusableResponseError with a content-free reason."""
    feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(feedback, "block_reason", None)
    if block_reason:
        raise UnusableResponseError(f"prompt blocked ({_enum_name(block_reason)})")
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        raise UnusableResponseError("no candidates")
    finish_reason = _enum_name(getattr(candidates[0], "finish_reason", None))
    # MAX_TOKENS means truncated JSON; SAFETY, RECITATION, SPII... mean partial or no output.
    if finish_reason not in (None, "STOP"):
        raise UnusableResponseError(f"finish reason {finish_reason}")

    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, schema_cls):
        return parsed
    try:
        if parsed is not None:
            return schema_cls.model_validate(parsed)
        text = _response_text(response)
        if not text:
            raise UnusableResponseError("empty response")
        if len(text) > _MAX_RESPONSE_CHARS:
            raise UnusableResponseError("response too large")
        return schema_cls.model_validate_json(text)
    except ValidationError as exc:
        # The ValidationError text quotes the response (personal data), so only
        # the error types are kept.
        error_types = sorted({str(e.get("type")) for e in exc.errors(include_url=False, include_input=False)})
        raise UnusableResponseError(f"invalid JSON ({', '.join(error_types)})") from None


def _response_text(response: Any) -> str | None:
    try:
        text = response.text
    except Exception:  # the SDK raises for some multi-candidate or non-text responses
        return None
    return text if isinstance(text, str) else None


def _usage(response: Any) -> dict[str, int]:
    metadata = getattr(response, "usage_metadata", None)
    usage: dict[str, int] = {}
    for key, attribute in _USAGE_FIELDS.items():
        count = getattr(metadata, attribute, None)
        if isinstance(count, int) and not isinstance(count, bool):
            usage[key] = count
    return usage


def _translate_error(exc: Exception) -> PdfWizardError:
    """Maps SDK, auth, and network errors to errors with safe, actionable user messages."""
    if isinstance(exc, PdfWizardError):
        return exc
    if isinstance(exc, auth_exceptions.DefaultCredentialsError):
        return ConfigError(CREDENTIALS_NOT_FOUND)
    if isinstance(exc, auth_exceptions.RefreshError):
        return ConfigError(CREDENTIALS_INVALID)
    if isinstance(exc, genai_errors.APIError):
        return MappingError(_api_error_message(_status_code(exc)))
    if isinstance(exc, (httpx.TimeoutException, auth_exceptions.TimeoutError)):
        return MappingError(
            "The AI model didn't respond in time. Try again, or try fewer or smaller documents."
        )
    if isinstance(exc, (httpx.TransportError, auth_exceptions.TransportError)):
        return MappingError("Couldn't reach the AI service. Check your internet connection and try again.")
    return MappingError("The request to the AI model failed. Try again later.")


def _api_error_message(code: int | None) -> str:
    status = f" (HTTP {code})" if code else ""
    if code in (401, 403):
        return (
            f"The AI service denied access{status}. Check that the Vertex AI API is enabled for your "
            "project and that your account can use it."
        )
    if code == 404:
        return f"The AI model wasn't found{status}. Check PDF_WIZARD_MODEL and GOOGLE_CLOUD_LOCATION."
    if code == 429:
        return f"The AI service is busy or your quota is used up{status}. Wait a minute and try again."
    if code is not None and code >= 500:
        return f"The AI service had a temporary problem{status}. Try again later."
    if code == 400:
        return f"The AI service rejected the request{status}. Try fewer or smaller documents."
    return f"The AI service returned an error{status}. Try again later."


def _status_code(exc: BaseException) -> int | None:
    code = getattr(exc, "code", None) if isinstance(exc, genai_errors.APIError) else None
    return code if isinstance(code, int) and not isinstance(code, bool) else None


def _enum_name(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)
