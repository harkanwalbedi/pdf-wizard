"""Runtime settings, read from environment variables.

Secrets (e.g. GEMINI_API_KEY) are never stored on these objects and never get
default values; they are read only where a client is created.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from .errors import ConfigError

MB = 1024 * 1024

# Model IDs change over time; override with PDF_WIZARD_MODEL. Pick the model
# (and the confidence thresholds) with the eval harness in evals/.
DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_LOCATION = "global"


@dataclass(frozen=True)
class Limits:
    """Caps that bound the work done on untrusted input."""

    max_file_bytes: int = 10 * MB
    max_total_bytes: int = 20 * MB  # stays within Gemini inline request limits
    max_source_files: int = 10
    max_pages: int = 100
    max_fields: int = 1000
    max_source_chars: int = 200_000
    max_note_chars: int = 5_000
    max_json_keys: int = 5_000
    max_json_depth: int = 32


@dataclass(frozen=True)
class Settings:
    backend: str = "vertex"  # "vertex" (recommended for personal data) or "gemini-api"
    project: str | None = None
    location: str = DEFAULT_LOCATION
    model: str = DEFAULT_MODEL
    # None = use the model's default temperature. Gemini 3 guidance is to keep
    # the default; structured output enforces the response format regardless.
    temperature: float | None = None
    seed: int | None = 7
    min_confidence: float = 0.7
    min_confidence_unverifiable: float = 0.85
    request_timeout_s: float = 180.0
    include_form_visual: bool = True
    limits: Limits = field(default_factory=Limits)
    # Directory for AI-enriched schemas of blank forms (no user data). None = no disk cache.
    schema_cache_dir: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        backend = env.get("PDF_WIZARD_BACKEND", "vertex").strip().lower()
        if backend not in ("vertex", "gemini-api"):
            raise ConfigError("PDF_WIZARD_BACKEND must be 'vertex' or 'gemini-api'.")
        min_conf = _float(env, "PDF_WIZARD_MIN_CONFIDENCE", 0.7, 0.0, 1.0)
        min_conf_unverifiable = _float(
            env, "PDF_WIZARD_MIN_CONFIDENCE_UNVERIFIABLE", max(0.85, min_conf), 0.0, 1.0
        )
        temperature_raw = env.get("PDF_WIZARD_TEMPERATURE", "").strip()
        return cls(
            backend=backend,
            project=(env.get("GOOGLE_CLOUD_PROJECT") or "").strip() or None,
            location=(env.get("GOOGLE_CLOUD_LOCATION") or DEFAULT_LOCATION).strip(),
            model=(env.get("PDF_WIZARD_MODEL") or DEFAULT_MODEL).strip(),
            temperature=_float(env, "PDF_WIZARD_TEMPERATURE", 0.0, 0.0, 2.0) if temperature_raw else None,
            min_confidence=min_conf,
            min_confidence_unverifiable=max(min_conf_unverifiable, min_conf),
            request_timeout_s=_float(env, "PDF_WIZARD_TIMEOUT_S", 180.0, 10.0, 900.0),
            schema_cache_dir=_schema_cache_dir(env),
        )

    def validate_for_model_calls(self, env: Mapping[str, str] | None = None) -> None:
        """Fail closed if the AI backend isn't configured."""
        env = os.environ if env is None else env
        if not self.model:
            raise ConfigError("Set PDF_WIZARD_MODEL to a Gemini model ID.")
        if self.backend == "vertex" and not self.project:
            raise ConfigError(
                "Set GOOGLE_CLOUD_PROJECT to a Google Cloud project with Vertex AI enabled, "
                "and run 'gcloud auth application-default login'."
            )
        if self.backend == "gemini-api" and not env.get("GEMINI_API_KEY"):
            raise ConfigError("Set GEMINI_API_KEY (paid tier only), or use the default Vertex AI backend.")


def _schema_cache_dir(env: Mapping[str, str]) -> str | None:
    """PDF_WIZARD_SCHEMA_CACHE=0 disables; PDF_WIZARD_SCHEMA_CACHE_DIR overrides the location."""
    if env.get("PDF_WIZARD_SCHEMA_CACHE", "1").strip() == "0":
        return None
    explicit = (env.get("PDF_WIZARD_SCHEMA_CACHE_DIR") or "").strip()
    if explicit:
        return os.path.expanduser(explicit)
    base = (env.get("XDG_CACHE_HOME") or "").strip() or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "pdf-wizard", "schemas")


def _float(env: Mapping[str, str], name: str, default: float, lo: float, hi: float) -> float:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number.") from exc
    if not lo <= value <= hi:
        raise ConfigError(f"{name} must be between {lo} and {hi}.")
    return value


def legacy_heuristics_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Phase 0 measurement flag (temporary): PDF_WIZARD_LEGACY_HEURISTICS=0 turns off the
    form-specific label/section regex heuristics so their value can be measured."""
    env = os.environ if env is None else env
    return env.get("PDF_WIZARD_LEGACY_HEURISTICS", "1").strip() != "0"
