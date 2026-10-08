"""Accuracy eval harness (design doc, Section 10).

A case is a directory under ``evals/cases/``::

    <name>/
        form.pdf        the blank target form
        case.json       sources and options (see `load_case`)
        expected.json   field name -> expected value; null means "must stay blank"
        sources/        files referenced by case.json

Every writable field of the form is scored. Fields that expected.json doesn't
list must stay blank, so a model that invents values is caught even on fields
the case author didn't think about.

Metrics, micro-averaged over fields:

- precision: correct fills / all fills
- recall: correct fills / fields that should be filled
- hallucination rate: fills on fields that should stay blank / fields that should stay blank

Values are read back from the filled PDF, so the scores reflect what the user
actually gets. The model is called once per case, model, and repeat; its answer
is replayed for every confidence threshold, so a threshold sweep is free.
"""

from __future__ import annotations

import json
import logging
import os
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, astuple, dataclass, field, replace
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pdf_wizard import __version__
from pdf_wizard.config import Settings
from pdf_wizard.errors import PdfWizardError
from pdf_wizard.mapping.base import Mapper, MappingRequest
from pdf_wizard.models import FieldStatus, FieldType, FormField, FormInfo, MappingResult
from pdf_wizard.pdf.inspector import open_form
from pdf_wizard.pipeline import FillOptions, FillOutcome, fill_form
from pdf_wizard.sources import SourceInput

log = logging.getLogger(__name__)

CASES_DIR = Path(__file__).resolve().parent / "cases"
FORM_FILE, CASE_FILE, EXPECTED_FILE = "form.pdf", "case.json", "expected.json"
ROLES = ("data", "doc", "reference", "note")
# Expected value meaning "this checkbox is checked", whatever its on-state is called.
CHECKED = "checked"
_UNCHECKED = frozenset({"off", "unchecked"})


class CaseError(Exception):
    """An eval case is malformed. The message says what to fix."""


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CaseSource:
    role: str  # one of ROLES
    path: Path | None = None  # files
    text: str | None = field(default=None, repr=False)  # notes
    row: int | None = None  # data files: 1-based record

    def to_input(self) -> SourceInput:
        if self.role == "note":
            return SourceInput(role="note", name="", content=(self.text or "").encode("utf-8"))
        assert self.path is not None
        return SourceInput(role=self.role, name=self.path.name, content=self.path.read_bytes(), row=self.row)


@dataclass(frozen=True)
class EvalCase:
    name: str
    form_path: Path
    sources: tuple[CaseSource, ...]
    # Field name -> accepted values (any one is correct); () means "must stay blank".
    expected: dict[str, tuple[str, ...]] = field(repr=False)
    today: date | None = None
    description: str = ""

    def inputs(self) -> list[SourceInput]:
        return [s.to_input() for s in self.sources]


def load_case(case_dir: Path) -> EvalCase:
    """Reads and validates one case directory.

    case.json::

        {
          "description": "What the case tests.",
          "today": "2026-10-06",                       # optional; fixes S0's date
          "sources": [
            {"role": "data", "file": "sources/applicant.json"},
            {"role": "data", "file": "sources/people.csv", "row": 2},
            {"role": "doc", "file": "sources/letter.eml"},
            {"role": "reference", "file": "sources/last_year.pdf"},
            {"role": "note", "text": "Use my work address."}
          ]
        }

    expected.json maps field names to null (must stay blank), a string, a list
    of acceptable strings, or true/false for checkboxes ("checked"/"unchecked"
    also work). Comparison ignores case and repeated whitespace; list format
    variants (e.g. phone numbers) explicitly.
    """
    case_dir = case_dir.resolve()
    name = case_dir.name
    if not (case_dir / FORM_FILE).is_file():
        raise CaseError(f"{name}: missing {FORM_FILE}")
    spec = _read_json(case_dir / CASE_FILE, name)
    expected = _read_json(case_dir / EXPECTED_FILE, name)
    if not isinstance(spec, dict):
        raise CaseError(f"{name}: {CASE_FILE} must contain a JSON object")
    if not isinstance(expected, dict):
        raise CaseError(f"{name}: {EXPECTED_FILE} must contain a JSON object")
    raw_sources = spec.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise CaseError(f"{name}: {CASE_FILE} needs a non-empty 'sources' list")
    return EvalCase(
        name=name,
        form_path=case_dir / FORM_FILE,
        sources=tuple(_parse_source(case_dir, f"{name}: source {i}", s) for i, s in enumerate(raw_sources, 1)),
        expected={str(k): _parse_expected(v, f"{name}: {EXPECTED_FILE}[{k!r}]") for k, v in expected.items()},
        today=_parse_date(spec.get("today"), name),
        description=str(spec.get("description") or ""),
    )


def discover_cases(root: Path = CASES_DIR, names: Sequence[str] = ()) -> list[EvalCase]:
    """Loads every case under `root`, or only the named ones (in the given order)."""
    if not root.is_dir():
        raise CaseError(f"cases directory not found: {root}")
    dirs = {p.name: p for p in sorted(root.iterdir()) if p.is_dir() and (p / CASE_FILE).is_file()}
    if names:
        missing = [n for n in names if n not in dirs]
        if missing:
            raise CaseError(f"unknown case(s): {', '.join(missing)}")
        selected = [dirs[n] for n in dict.fromkeys(names)]
    else:
        selected = list(dirs.values())
    if not selected:
        raise CaseError(f"no cases found in {root}")
    return [load_case(p) for p in selected]


def check_expected(case: EvalCase, form: FormInfo) -> None:
    """Raises CaseError if expected.json can't be scored against this form."""
    writable = [f.name for f in form.fields if f.writable]
    duplicates = sorted(n for n, count in Counter(writable).items() if count > 1)
    if duplicates:
        raise CaseError(f"{case.name}: the form has duplicate field names: {', '.join(duplicates)}")
    unknown = sorted(set(case.expected) - set(writable))
    if unknown:
        raise CaseError(
            f"{case.name}: {EXPECTED_FILE} names fields that aren't writable fields of the form: "
            + ", ".join(unknown)
        )


def validate_cases(cases: Iterable[EvalCase], settings: Settings) -> None:
    """Checks every case against its form before any model is called."""
    for case in cases:
        check_expected(case, _open_case_form(case, settings))
        for source in case.sources:
            if source.path is not None and source.path.stat().st_size > settings.limits.max_file_bytes:
                raise CaseError(f"{case.name}: {source.path.name} is larger than the file size limit")


def _open_case_form(case: EvalCase, settings: Settings) -> FormInfo:
    try:
        return open_form(case.form_path.read_bytes(), limits=settings.limits).info
    except PdfWizardError as exc:
        raise CaseError(f"{case.name}: {FORM_FILE}: {exc.user_message}") from exc


def _read_json(path: Path, name: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CaseError(f"{name}: missing {path.name}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CaseError(f"{name}: {path.name} is not valid UTF-8 JSON ({exc})") from exc


def _parse_source(case_dir: Path, where: str, raw: Any) -> CaseSource:
    if not isinstance(raw, dict):
        raise CaseError(f"{where} must be a JSON object")
    role = raw.get("role")
    if role not in ROLES:
        raise CaseError(f"{where}: 'role' must be one of {', '.join(ROLES)}")
    if role == "note":
        text = raw.get("text")
        if not isinstance(text, str) or not text.strip():
            raise CaseError(f"{where}: a note needs a non-empty 'text'")
        return CaseSource(role, text=text)
    file = raw.get("file")
    if not isinstance(file, str) or not file:
        raise CaseError(f"{where}: needs a 'file' path relative to the case directory")
    path = (case_dir / file).resolve()
    if not path.is_relative_to(case_dir):
        raise CaseError(f"{where}: 'file' must stay inside the case directory")
    if not path.is_file():
        raise CaseError(f"{where}: {file} not found")
    row = raw.get("row")
    if row is not None and (role != "data" or not isinstance(row, int) or isinstance(row, bool) or row < 1):
        raise CaseError(f"{where}: 'row' must be a positive integer, and only on data sources")
    return CaseSource(role, path=path, row=row)


def _parse_expected(value: Any, where: str) -> tuple[str, ...]:
    if value is None or value is False:
        return ()
    if value is True:
        return (CHECKED,)
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return tuple(v for v in value if v.strip())
    raise CaseError(f"{where} must be null, true/false, a string, or a list of strings")


def _parse_date(value: Any, name: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise CaseError(f"{name}: 'today' must be an ISO date such as 2026-10-06") from exc


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


class Outcome(StrEnum):
    CORRECT = "correct"  # filled with an accepted value
    WRONG = "wrong"  # filled, but not with an accepted value
    HALLUCINATED = "hallucinated"  # filled, but should have stayed blank
    MISSED = "missed"  # left blank, but a value was expected
    CORRECT_BLANK = "correct_blank"  # left blank, as expected


@dataclass(frozen=True)
class FieldScore:
    name: str
    outcome: Outcome
    status: str | None = None  # FieldStatus from the fill report
    reason: str | None = None  # RejectReason, for rejected proposals
    # Values are kept for `--show-values` only. They never go into results files.
    expected: tuple[str, ...] = field(default=(), repr=False)
    got: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class Metrics:
    # Field names match the Outcome values.
    correct: int = 0
    wrong: int = 0
    hallucinated: int = 0
    missed: int = 0
    correct_blank: int = 0

    @classmethod
    def of(cls, scores: Iterable[FieldScore]) -> Metrics:
        counts = Counter(s.outcome for s in scores)
        return cls(**{o.value: counts.get(o, 0) for o in Outcome})

    def __add__(self, other: Metrics) -> Metrics:
        return Metrics(*(a + b for a, b in zip(astuple(self), astuple(other), strict=True)))

    @property
    def filled(self) -> int:
        return self.correct + self.wrong + self.hallucinated

    @property
    def should_fill(self) -> int:
        return self.correct + self.wrong + self.missed

    @property
    def should_stay_blank(self) -> int:
        return self.hallucinated + self.correct_blank

    @property
    def precision(self) -> float | None:
        return _ratio(self.correct, self.filled)

    @property
    def recall(self) -> float | None:
        return _ratio(self.correct, self.should_fill)

    @property
    def hallucination_rate(self) -> float | None:
        return _ratio(self.hallucinated, self.should_stay_blank)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "precision": self.precision,
            "recall": self.recall,
            "hallucination_rate": self.hallucination_rate,
        }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def score_outcome(case: EvalCase, form: FormInfo, outcome: FillOutcome | None, *, settings: Settings) -> list[FieldScore]:
    """Scores every writable field. `outcome` is None when the pipeline failed (nothing filled)."""
    results = {r.name: r for r in outcome.report.fields} if outcome is not None else {}
    written: dict[str, str | None] = {}
    if outcome is not None and outcome.pdf is not None:
        refilled = open_form(outcome.pdf, limits=settings.limits, for_filling=False).info
        written = {f.name: f.current_value for f in refilled.fields}

    scores = []
    for f in form.fields:
        if not f.writable:
            continue
        result = results.get(f.name)
        filled = result is not None and result.status is FieldStatus.FILLED
        got = _blank_to_none(f, written.get(f.name)) if filled else None
        accepted = _accepted_values(f, case.expected.get(f.name, ()))
        scores.append(
            FieldScore(
                name=f.name,
                outcome=_classify(f, got, accepted),
                status=result.status.value if result is not None else None,
                reason=result.reason.value if result is not None and result.reason is not None else None,
                expected=accepted,
                got=got,
            )
        )
    return scores


def _blank_to_none(f: FormField, value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    if f.type in (FieldType.CHECKBOX, FieldType.RADIO) and value == "Off":
        return None
    return value


def _accepted_values(f: FormField, accepted: tuple[str, ...]) -> tuple[str, ...]:
    if f.type in (FieldType.CHECKBOX, FieldType.RADIO):
        return tuple(a for a in accepted if _norm(a) not in _UNCHECKED)  # unchecked == blank
    return accepted


def _classify(f: FormField, got: str | None, accepted: tuple[str, ...]) -> Outcome:
    if got is None:
        return Outcome.MISSED if accepted else Outcome.CORRECT_BLANK
    if not accepted:
        return Outcome.HALLUCINATED
    return Outcome.CORRECT if _matches(f, got, accepted) else Outcome.WRONG


def _matches(f: FormField, got: str, accepted: tuple[str, ...]) -> bool:
    wanted = {_norm(a) for a in accepted}
    if f.type is FieldType.CHECKBOX and (len(f.on_states) <= 1 or CHECKED in wanted):
        return True  # `got` is an on-state, and a value (rather than blank) was expected
    candidates = {_norm(got)}
    if f.type in (FieldType.RADIO, FieldType.CHOICE):
        candidates |= {_norm(o.label) for o in f.options if o.value == got and o.label}  # accept labels too
    return not candidates.isdisjoint(wanted)


def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, order=True)
class RunConfig:
    model: str
    threshold: float

    @property
    def label(self) -> str:
        return f"{self.model} @ {self.threshold:.2f}"


@dataclass
class CaseRun:
    case: str
    config: RunConfig
    scores: list[FieldScore]
    repeat: int = 1
    error: str | None = None  # safe message when the pipeline failed
    prompt_version: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    mapping_ms: int | None = None  # model latency; only on the run that called the model
    schema_calls: int = 0  # schema-enrichment model calls made for this case (0 when cached or off)
    schema_ms: int | None = None  # inspect + schema stage time; only on the first run

    @property
    def metrics(self) -> Metrics:
        return Metrics.of(self.scores)


class ReplayMapper:
    """Calls the wrapped mapper once, then replays its result (or its error)."""

    def __init__(self, inner: Mapper):
        self._inner = inner
        self._called = False
        self._result: MappingResult | None = None
        self._error: Exception | None = None
        self.schema_calls = 0

    def propose(self, request: MappingRequest) -> MappingResult:
        if not self._called:
            self._called = True
            try:
                self._result = self._inner.propose(request)
            except Exception as exc:
                self._error = exc
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result

    def enrich_schema(self, schema: Any, form_pdf: bytes) -> Any:
        # Results are cached by schema_cache, so this runs at most once per form per process.
        self.schema_calls += 1
        return self._inner.enrich_schema(schema, form_pdf)  # type: ignore[attr-defined]


def _eval_enrich_schema() -> bool:
    """Phase 0 measurement switch: PDF_WIZARD_EVAL_ENRICH_SCHEMA=1 runs visual schema enrichment."""
    return os.environ.get("PDF_WIZARD_EVAL_ENRICH_SCHEMA", "0").strip() == "1"


def run_case(
    case: EvalCase,
    mapper: Mapper,
    *,
    model: str,
    thresholds: Sequence[float],
    settings: Settings,
    repeat: int = 1,
) -> list[CaseRun]:
    """Runs one case with one mapper, once per threshold, calling the model only once."""
    form_bytes = case.form_path.read_bytes()
    inputs = case.inputs()
    form = _open_case_form(case, settings)
    check_expected(case, form)
    replay = ReplayMapper(mapper)
    runs = []
    for index, threshold in enumerate(thresholds):
        outcome: FillOutcome | None = None
        error: str | None = None
        calls_before = replay.schema_calls
        try:
            outcome = fill_form(
                form_bytes,
                inputs,
                mapper=replay,
                settings=settings,
                options=FillOptions(min_confidence=threshold, enrich_schema=_eval_enrich_schema()),
                form_name=f"{case.name}.pdf",
                today=case.today,
            )
        except PdfWizardError as exc:
            error = exc.user_message
        except Exception as exc:  # a bug; record it so one case can't sink the whole run
            log.error("case %s crashed: %s", case.name, type(exc).__name__)
            log.debug("traceback for case %s", case.name, exc_info=True)
            error = f"Internal error ({type(exc).__name__})."
        report = outcome.report if outcome is not None else None
        runs.append(
            CaseRun(
                case=case.name,
                config=RunConfig(model, threshold),
                scores=score_outcome(case, form, outcome, settings=settings),
                repeat=repeat,
                error=error,
                prompt_version=report.prompt_version if report else None,
                usage=dict(report.usage) if report and index == 0 else {},
                mapping_ms=report.timings_ms.get("mapping") if report and index == 0 else None,
                schema_calls=replay.schema_calls - calls_before,
                schema_ms=report.timings_ms.get("inspect") if report and index == 0 else None,
            )
        )
    return runs


@dataclass
class SuiteResult:
    runs: list[CaseRun]
    models: list[str]
    thresholds: list[float]
    repeats: int = 1
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def configs(self) -> list[RunConfig]:
        return [RunConfig(m, t) for m in self.models for t in self.thresholds]

    def runs_for(self, config: RunConfig) -> list[CaseRun]:
        return sorted((r for r in self.runs if r.config == config), key=lambda r: (r.case, r.repeat))

    def metrics(self, config: RunConfig) -> Metrics:
        return sum((r.metrics for r in self.runs_for(config)), Metrics())

    def errors(self, config: RunConfig) -> int:
        return sum(1 for r in self.runs_for(config) if r.error)

    def model_stats(self, model: str) -> dict[str, Any]:
        """Calls, errors, mean latency, and tokens: from the runs that called the model."""
        calls = [r for r in self.runs if r.config == RunConfig(model, self.thresholds[0])]
        latencies = [r.mapping_ms for r in calls if r.mapping_ms is not None]
        tokens: Counter[str] = Counter()
        for r in calls:
            tokens.update(r.usage)
        schema_ms = [r.schema_ms for r in calls if r.schema_ms is not None]
        return {
            "calls": len(calls),
            "errors": sum(1 for r in calls if r.error),
            "mean_latency_ms": round(sum(latencies) / len(latencies)) if latencies else None,
            "tokens": dict(tokens),
            "schema_calls": sum(r.schema_calls for r in self.runs if r.config.model == model),
            "mean_schema_ms": round(sum(schema_ms) / len(schema_ms)) if schema_ms else None,
        }

    def recommend(self, target_precision: float) -> RunConfig | None:
        """The config with the best recall among those meeting the precision target.

        Ties go to higher precision, then to the model listed first (list the
        cheaper model first), then to the higher (safer) threshold.
        """
        best: RunConfig | None = None
        best_key: tuple | None = None
        for config in self.configs():
            m = self.metrics(config)
            precision = 1.0 if m.precision is None else m.precision  # filled nothing: nothing wrong
            if precision < target_precision:
                continue
            key = (m.recall or 0.0, precision, -self.models.index(config.model), config.threshold)
            if best_key is None or key > best_key:
                best, best_key = config, key
        return best

    def to_json(self, *, target_precision: float) -> dict[str, Any]:
        """Results without any field values, safe to keep next to the code."""
        recommended = self.recommend(target_precision)
        return {
            "tool_version": __version__,
            "created_at": self.created_at,
            "prompt_versions": sorted({r.prompt_version for r in self.runs if r.prompt_version}),
            "cases": sorted({r.case for r in self.runs}),
            "repeats": self.repeats,
            "target_precision": target_precision,
            "recommended": asdict(recommended) if recommended else None,
            "models": {m: self.model_stats(m) for m in self.models},
            "configs": [
                {"model": c.model, "threshold": c.threshold, "errors": self.errors(c), **self.metrics(c).to_dict()}
                for c in self.configs()
            ],
            "runs": [
                {
                    "case": r.case,
                    "model": r.config.model,
                    "threshold": r.config.threshold,
                    "repeat": r.repeat,
                    "error": r.error,
                    **r.metrics.to_dict(),
                    "fields": [
                        {"name": s.name, "outcome": s.outcome.value, "status": s.status, "reason": s.reason}
                        for s in r.scores
                    ],
                }
                for c in self.configs()
                for r in self.runs_for(c)
            ],
        }


def run_suite(
    cases: Sequence[EvalCase],
    *,
    models: Sequence[str],
    thresholds: Sequence[float],
    settings: Settings,
    mapper_factory: Callable[[str], Mapper],
    repeats: int = 1,
    jobs: int = 4,
    on_done: Callable[[EvalCase, str, int], None] | None = None,
) -> SuiteResult:
    """Runs every case with every model (`repeats` times each) and every threshold."""
    if not cases or not models or not thresholds:
        raise ValueError("need at least one case, model, and threshold")
    models = list(dict.fromkeys(models))
    thresholds = sorted(set(thresholds))
    validate_cases(cases, settings)  # fail before spending money on model calls

    def work(case: EvalCase, model: str, repeat: int) -> list[CaseRun]:
        return run_case(
            case,
            mapper_factory(model),
            model=model,
            thresholds=thresholds,
            settings=replace(settings, model=model),
            repeat=repeat,
        )

    tasks = [(case, model, repeat) for model in models for case in cases for repeat in range(1, repeats + 1)]
    runs: list[CaseRun] = []
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {pool.submit(work, *task): task for task in tasks}
        for future in as_completed(futures):
            runs.extend(future.result())
            if on_done is not None:
                on_done(*futures[future])
    return SuiteResult(runs=runs, models=models, thresholds=thresholds, repeats=repeats)
