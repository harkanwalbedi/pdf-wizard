"""Runs the accuracy evals against live Gemini models.

    python -m evals.run                                  # default model, threshold sweep
    python -m evals.run --model gemini-3.8-flash --model gemini-2.5-pro --repeat 3
    python -m evals.run --case w9_from_email --show-values --out evals/results/latest.json

Needs the same configuration as `pdf-wizard fill`: GOOGLE_CLOUD_PROJECT and
Application Default Credentials for Vertex AI. Every case is sent to every
model, so a run makes (cases x models x repeats) paid model calls. Thresholds
cost nothing extra: each model answer is replayed for every threshold.

Exit codes: 0 = a config met the precision target, 1 = none did,
2 = a case is malformed, 4 = configuration error.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TextIO

from pdf_wizard.config import Settings
from pdf_wizard.errors import PdfWizardError
from pdf_wizard.logsafe import configure_logging
from pdf_wizard.mapping.base import Mapper

from .harness import CASES_DIR, CaseError, EvalCase, Outcome, RunConfig, SuiteResult, discover_cases, run_suite

DEFAULT_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
DEFAULT_TARGET_PRECISION = 0.98
_FAILURES = (Outcome.WRONG, Outcome.HALLUCINATED, Outcome.MISSED)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m evals.run", description="Measure PDF Wizard's accuracy on eval cases.")
    p.add_argument("--cases", type=Path, default=CASES_DIR, metavar="DIR", help="cases directory (default: evals/cases)")
    p.add_argument("--case", action="append", default=[], metavar="NAME", help="run only this case (repeatable)")
    p.add_argument(
        "--model", action="append", default=[], metavar="ID",
        help="model to evaluate (repeatable; default: PDF_WIZARD_MODEL). List the cheaper model first: it wins ties.",
    )
    p.add_argument(
        "--threshold", action="append", type=_unit_float, default=[], metavar="T",
        help="confidence threshold to try (repeatable; default: 0.5 0.6 0.7 0.8 0.9)",
    )
    p.add_argument(
        "--target-precision", type=_unit_float, default=DEFAULT_TARGET_PRECISION, metavar="P",
        help="precision a config needs to be recommended (default: 0.98)",
    )
    p.add_argument("--repeat", type=_bounded_int(1, 10), default=1, metavar="N", help="run each case N times (default: 1)")
    p.add_argument("--jobs", type=_bounded_int(1, 16), default=4, metavar="N", help="cases run in parallel (default: 4)")
    p.add_argument("--out", type=Path, metavar="FILE", help="also write the results as JSON (no field values)")
    p.add_argument(
        "--show-values", action="store_true",
        help="print expected and actual values of failed fields (may print personal data)",
    )
    p.add_argument("--verbose", action="store_true", help="per-case details for every config")
    p.add_argument("--debug", action="store_true", help="log tracebacks of crashed cases")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    return run(argv)


def run(
    argv: Sequence[str] | None = None,
    *,
    mapper_factory: Callable[[str], Mapper] | None = None,
    settings: Settings | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Entry point. Tests inject `mapper_factory` (model ID -> Mapper) to avoid the network."""
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    args = build_parser().parse_args(argv)
    configure_logging("DEBUG" if args.debug else "WARNING")
    try:
        settings = settings or Settings.from_env()
        models = args.model or [settings.model]
        thresholds = args.threshold or list(DEFAULT_THRESHOLDS)
        cases = discover_cases(args.cases, args.case)
        if mapper_factory is None:
            mapper_factory = _gemini_factory(settings, models)
            calls = len(cases) * len(set(models)) * args.repeat
            print(f"Sending {len(cases)} case(s) to {len(set(models))} model(s), {args.repeat} time(s) each: "
                  f"{calls} model call(s).", file=stderr)

        total = len(cases) * len(set(models)) * args.repeat
        done = 0

        def progress(case: EvalCase, model: str, repeat: int) -> None:
            nonlocal done
            done += 1
            print(f"  [{done}/{total}] {case.name} with {model}" + (f" (#{repeat})" if args.repeat > 1 else ""),
                  file=stderr)

        result = run_suite(
            cases,
            models=models,
            thresholds=thresholds,
            settings=settings,
            mapper_factory=mapper_factory,
            repeats=args.repeat,
            jobs=args.jobs,
            on_done=progress,
        )
    except CaseError as exc:
        print(f"Error: {exc}", file=stderr)
        return 2
    except PdfWizardError as exc:
        print(f"Error: {exc.user_message}", file=stderr)
        return exc.exit_code

    print(format_report(result, target_precision=args.target_precision, default_threshold=settings.min_confidence,
                        show_values=args.show_values, verbose=args.verbose), file=stdout)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result.to_json(target_precision=args.target_precision), indent=2) + "\n",
                            encoding="utf-8")
        print(f"Results written to {args.out}", file=stderr)
    return 0 if result.recommend(args.target_precision) else 1


def _gemini_factory(settings: Settings, models: Sequence[str]) -> Callable[[str], Mapper]:
    from pdf_wizard.mapping.gemini import create_mapper  # imported lazily: tests never need the SDK

    def factory(model: str) -> Mapper:
        return create_mapper(replace(settings, model=model))

    for model in models:  # fail on configuration errors before the first model call
        factory(model)
    return factory


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def format_report(
    result: SuiteResult,
    *,
    target_precision: float,
    default_threshold: float = 0.7,
    show_values: bool = False,
    verbose: bool = False,
) -> str:
    cases = sorted({r.case for r in result.runs})
    lines = [f"PDF Wizard evals: {len(cases)} case(s), {len(result.models)} model(s), {result.repeats} repeat(s) each"]
    versions = sorted({r.prompt_version for r in result.runs if r.prompt_version})
    if versions:
        lines.append(f"Prompt version: {', '.join(versions)}")
    lines.append("")

    width = max(5, *(len(m) for m in result.models))
    lines.append(
        f"{'model':<{width}}  {'thresh':>6}  {'precision':>9}  {'recall':>7}  {'halluc.rate':>11}"
        f"  {'correct':>7}  {'wrong':>5}  {'hallucinated':>12}  {'missed':>6}  {'errors':>6}"
    )
    for config in result.configs():
        m = result.metrics(config)
        lines.append(
            f"{config.model:<{width}}  {config.threshold:>6.2f}  {_pct(m.precision):>9}  {_pct(m.recall):>7}"
            f"  {_pct(m.hallucination_rate):>11}  {m.correct:>7}  {m.wrong:>5}  {m.hallucinated:>12}"
            f"  {m.missed:>6}  {result.errors(config):>6}"
        )

    lines += ["", "Model calls:"]
    for model in result.models:
        stats = result.model_stats(model)
        latency = f"{stats['mean_latency_ms'] / 1000:.1f} s" if stats["mean_latency_ms"] is not None else "n/a"
        tokens = ", ".join(f"{k} {v:,}" for k, v in sorted(stats["tokens"].items())) or "n/a"
        lines.append(f"  {model}: {stats['calls']} call(s), {stats['errors']} error(s), "
                     f"mean latency {latency}, tokens: {tokens}")
        if stats.get("schema_calls"):
            schema_ms = stats.get("mean_schema_ms")
            schema_s = f"{schema_ms / 1000:.1f} s" if schema_ms is not None else "n/a"
            lines.append(f"    + {stats['schema_calls']} schema-enrichment call(s), "
                         f"mean inspect+schema time per case {schema_s} (tokens not counted)")

    lines.append("")
    best = result.recommend(target_precision)
    if best:
        lines.append(f"Recommended: {best.model} at threshold {best.threshold:.2f} "
                     f"(best recall with precision >= {target_precision:.1%}).")
        lines.append(f"  export PDF_WIZARD_MODEL={best.model} PDF_WIZARD_MIN_CONFIDENCE={best.threshold:.2f}")
    else:
        lines.append(f"No configuration reached the precision target of {target_precision:.1%}.")

    focus = [best] if best else [_closest(result, default_threshold)]
    for config in result.configs() if verbose else focus:
        lines += ["", f"Per case, {config.label}:"]
        lines += _case_lines(result, config, show_values=show_values)
    return "\n".join(lines)


def _case_lines(result: SuiteResult, config: RunConfig, *, show_values: bool) -> list[str]:
    lines = []
    width = max(len(r.case) for r in result.runs) + (4 if result.repeats > 1 else 0)
    for run in result.runs_for(config):
        name = _safe(run.case) + (f" #{run.repeat}" if result.repeats > 1 else "")
        m = run.metrics
        lines.append(f"  {name:<{width}}  precision {_pct(m.precision):>6}  recall {_pct(m.recall):>6}"
                     f"  halluc.rate {_pct(m.hallucination_rate):>6}")
        if run.error:
            lines.append(f"      error: {_safe(run.error)}")
        for outcome in _FAILURES:
            failed = [s for s in run.scores if s.outcome is outcome]
            if not failed:
                continue
            if show_values:
                for s in failed:
                    expected = " | ".join(repr(_safe(v)) for v in s.expected) or "(blank)"
                    got = repr(_safe(s.got)) if s.got is not None else "(blank)"
                    lines.append(f"      {outcome.value}: {_safe(s.name)}{_why(s.status, s.reason)}: got {got}, "
                                 f"expected {expected}")
            else:
                names = ", ".join(_safe(s.name) + _why(s.status, s.reason) for s in failed)
                lines.append(f"      {outcome.value}: {names}")
    return lines


def _why(status: str | None, reason: str | None) -> str:
    if reason:
        return f" ({reason})"
    return f" ({status})" if status and status != "filled" else ""


def _closest(result: SuiteResult, threshold: float) -> RunConfig:
    return RunConfig(result.models[0], min(result.thresholds, key=lambda t: abs(t - threshold)))


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _safe(text: str, limit: int = 80) -> str:
    """Model output and documents are untrusted: drop control characters (terminal escapes)."""
    cleaned = "".join(ch if ch.isprintable() else " " for ch in text)
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1] + "…"


def _unit_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return value


def _bounded_int(lo: int, hi: int) -> Callable[[str], int]:
    def parse(raw: str) -> int:
        try:
            value = int(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be an integer") from exc
        if not lo <= value <= hi:
            raise argparse.ArgumentTypeError(f"must be between {lo} and {hi}")
        return value

    return parse


if __name__ == "__main__":
    raise SystemExit(main())
