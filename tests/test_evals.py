"""Tests for the eval harness (evals/) with fake mappers (no network)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

import pdf_factory as pf
from evals import run as eval_run
from evals.harness import (
    CHECKED,
    CaseError,
    Metrics,
    Outcome,
    RunConfig,
    discover_cases,
    load_case,
    run_case,
    run_suite,
    validate_cases,
)
from evals.harness import _classify, _accepted_values  # noqa: PLC2701 - rule-level tests
from pdf_wizard.config import Settings
from pdf_wizard.errors import MappingError
from pdf_wizard.mapping.base import MappingRequest
from pdf_wizard.mapping.fake import FakeMapper, proposal
from pdf_wizard.models import FieldType, FormField, Option

SSN = "123-45-6789"
APPLICANT = {
    "applicant": {
        "name": "Jane Q. Public",
        "ssn": SSN,
        "email": "jane@example.com",
        "married": True,
        "state": "TX",
        "previous_state": "NY",
        "job": "self-employed",
    }
}
# sample_form has 9 writable fields; unlisted ones (notes, color) must stay blank.
EXPECTED = {
    "full_name": "Jane Q. Public",
    "dob": "01/31/1990",
    "ssn": "123456789",
    "email": None,
    "married": True,
    "employment": "Self-employed",  # a label: matches the option value "self"
    "state": "TX",
}
PROPOSALS = {
    "full_name": ("Jane Q. Public", "applicant.name"),  # correct
    "email": ("jane@example.com", "applicant.email"),  # hallucinated: must stay blank
    "married": ("checked", "applicant.married"),  # correct
    "employment": ("self", "applicant.job"),  # correct (by label)
    "state": ("NY", "applicant.previous_state"),  # wrong
}  # dob and ssn are missed; notes and color stay blank


def name_mapper(spec: dict[str, tuple], *, confidence: float = 0.95, model: str = "fake-model") -> FakeMapper:
    def fn(request: MappingRequest):
        ids = {f.name: f.id for f in request.fields}
        return [
            proposal(ids[name], value, source_id="S1", evidence=evidence, confidence=confidence)
            for name, (value, evidence) in spec.items()
        ]

    return FakeMapper(fn=fn, model=model)


def make_case(root: Path, name: str = "sample", *, expected=None, sources=None, form: bytes | None = None,
              today: str | None = "2026-10-06") -> Path:
    case_dir = root / name
    (case_dir / "sources").mkdir(parents=True)
    (case_dir / "form.pdf").write_bytes(form or pf.sample_form())
    (case_dir / "sources" / "applicant.json").write_text(json.dumps(APPLICANT))
    spec = {"sources": sources or [{"role": "data", "file": "sources/applicant.json"}]}
    if today:
        spec["today"] = today
    (case_dir / "case.json").write_text(json.dumps(spec))
    (case_dir / "expected.json").write_text(json.dumps(EXPECTED if expected is None else expected))
    return case_dir


def by_name(scores):
    return {s.name: s for s in scores}


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def test_scores_every_outcome(tmp_path):
    case = load_case(make_case(tmp_path))
    [run] = run_case(case, name_mapper(PROPOSALS), model="fake", thresholds=[0.7], settings=Settings())

    scores = by_name(run.scores)
    assert "office_use" not in scores  # read-only fields aren't scored
    assert {n: s.outcome for n, s in scores.items()} == {
        "full_name": Outcome.CORRECT,
        "dob": Outcome.MISSED,
        "ssn": Outcome.MISSED,
        "email": Outcome.HALLUCINATED,
        "notes": Outcome.CORRECT_BLANK,
        "married": Outcome.CORRECT,
        "employment": Outcome.CORRECT,
        "state": Outcome.WRONG,
        "color": Outcome.CORRECT_BLANK,  # has a default value, but nothing was filled
    }
    assert scores["state"].got == "NY" and scores["state"].status == "filled"
    assert scores["married"].got == "Yes"  # values are read back from the filled PDF

    m = run.metrics
    assert (m.correct, m.wrong, m.hallucinated, m.missed, m.correct_blank) == (3, 1, 1, 2, 2)
    assert m.precision == pytest.approx(3 / 5)
    assert m.recall == pytest.approx(3 / 6)
    assert m.hallucination_rate == pytest.approx(1 / 3)
    assert run.error is None and run.prompt_version is not None


def test_rejected_proposals_are_blank_and_keep_their_reason(tmp_path):
    case = load_case(make_case(tmp_path))
    mapper = name_mapper({"full_name": ("John Smith", "John Smith")})  # evidence not in the source
    [run] = run_case(case, mapper, model="fake", thresholds=[0.7], settings=Settings())
    score = by_name(run.scores)["full_name"]
    assert score.outcome is Outcome.MISSED and score.status == "rejected" and score.reason == "unverified"


def test_threshold_sweep_calls_the_model_once(tmp_path):
    case = load_case(make_case(tmp_path))
    mapper = name_mapper({"full_name": ("Jane Q. Public", "applicant.name")}, confidence=0.75)
    runs = run_case(case, mapper, model="fake", thresholds=[0.5, 0.7, 0.8], settings=Settings())

    assert len(mapper.requests) == 1
    outcomes = [by_name(r.scores)["full_name"].outcome for r in runs]
    assert outcomes == [Outcome.CORRECT, Outcome.CORRECT, Outcome.MISSED]
    assert by_name(runs[2].scores)["full_name"].reason == "low_confidence"
    assert runs[0].mapping_ms is not None and runs[1].mapping_ms is None  # latency only where the model ran


def test_pipeline_errors_count_as_blank_and_are_reported(tmp_path):
    case = load_case(make_case(tmp_path))
    mapper = FakeMapper(error=MappingError("The AI model returned an unusable response."))
    runs = run_case(case, mapper, model="fake", thresholds=[0.5, 0.9], settings=Settings())

    assert len(mapper.requests) == 1  # the error is replayed, not retried
    assert all(r.error == "The AI model returned an unusable response." for r in runs)
    m = runs[0].metrics
    assert m.filled == 0 and m.missed == 6 and m.correct_blank == 3
    assert m.precision is None and m.recall == 0.0


def test_matching_rules():
    text = FormField(id="F1", name="t", type=FieldType.TEXT)
    assert _classify(text, "  JANE   q. public ", ("Jane Q. Public",)) is Outcome.CORRECT  # case and spacing
    assert _classify(text, "512-555-0147", ("(512) 555-0147", "512-555-0147")) is Outcome.CORRECT
    assert _classify(text, "Jane", ("Jane Q. Public",)) is Outcome.WRONG

    radio = FormField(id="F2", name="r", type=FieldType.RADIO, options=[Option(value="1", label="Individual")])
    assert _classify(radio, "1", ("individual",)) is Outcome.CORRECT  # labels count
    assert _classify(radio, None, _accepted_values(radio, ("Off",))) is Outcome.CORRECT_BLANK

    checkbox = FormField(id="F3", name="c", type=FieldType.CHECKBOX, options=[Option(value="On")])
    assert _classify(checkbox, "On", (CHECKED,)) is Outcome.CORRECT
    assert _classify(checkbox, "On", ("Yes",)) is Outcome.CORRECT  # single on-state: any value means checked
    assert _accepted_values(checkbox, ("unchecked",)) == ()  # unchecked means blank
    assert _classify(checkbox, "On", ()) is Outcome.HALLUCINATED

    pair = FormField(id="F4", name="p", type=FieldType.CHECKBOX, options=[Option(value="Yes"), Option(value="No")])
    assert _classify(pair, "No", ("Yes",)) is Outcome.WRONG  # several on-states: the state must match
    assert _classify(pair, "No", (CHECKED,)) is Outcome.CORRECT


def test_metrics_add_up():
    a = Metrics(correct=2, wrong=1)
    b = Metrics(missed=3, hallucinated=1, correct_blank=4)
    total = a + b
    assert (total.filled, total.should_fill, total.should_stay_blank) == (4, 6, 5)
    assert Metrics().precision is None and Metrics().recall is None


# --------------------------------------------------------------------------- #
# Suite, recommendation, and results
# --------------------------------------------------------------------------- #


def test_suite_recommends_the_most_accurate_config_and_hides_values(tmp_path):
    make_case(tmp_path, "a")
    make_case(tmp_path, "b")
    cases = discover_cases(tmp_path)
    good = {k: v for k, v in PROPOSALS.items() if k not in ("email", "state")}  # only correct values

    def factory(model: str):
        return name_mapper(good if model == "careful" else PROPOSALS, model=model)

    result = run_suite(cases, models=["sloppy", "careful"], thresholds=[0.9, 0.5], settings=Settings(),
                       mapper_factory=factory, jobs=2)

    assert result.thresholds == [0.5, 0.9]
    assert result.metrics(RunConfig("careful", 0.5)).precision == 1.0
    assert result.metrics(RunConfig("sloppy", 0.5)).precision == pytest.approx(0.6)
    # Same scores at both thresholds: the higher (safer) one wins the tie.
    assert result.recommend(0.98) == RunConfig("careful", 0.9)
    assert result.recommend(0.5) == RunConfig("careful", 0.9)  # better precision beats list order
    assert result.model_stats("careful")["calls"] == 2

    payload = json.dumps(result.to_json(target_precision=0.98))
    assert "Jane" not in payload and SSN not in payload and "jane@example.com" not in payload
    data = json.loads(payload)
    assert data["recommended"] == {"model": "careful", "threshold": 0.9}
    assert len(data["runs"]) == 2 * 2 * 2 and data["runs"][0]["fields"][0].keys() == {"name", "outcome", "status", "reason"}


def test_no_recommendation_when_precision_target_is_missed(tmp_path):
    cases = discover_cases(tmp_path) if make_case(tmp_path) else []
    result = run_suite(cases, models=["sloppy"], thresholds=[0.7], settings=Settings(),
                       mapper_factory=lambda model: name_mapper(PROPOSALS))
    assert result.recommend(0.98) is None


# --------------------------------------------------------------------------- #
# Case validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sources": [{"role": "data", "file": "../outside.json"}]}, "inside the case directory"),
        ({"sources": [{"role": "doc", "file": "sources/missing.txt"}]}, "not found"),
        ({"sources": [{"role": "note"}]}, "needs a non-empty 'text'"),
        ({"sources": [{"role": "doc", "file": "sources/applicant.json", "row": 2}]}, "'row'"),
        ({"sources": [{"role": "boss", "file": "sources/applicant.json"}]}, "'role'"),
        ({"expected": {"full_name": 42}}, "must be null, true/false, a string"),
        ({"today": "next tuesday"}, "ISO date"),
    ],
)
def test_malformed_cases_are_rejected_when_loaded(tmp_path, kwargs, message):
    (tmp_path / "outside.json").write_text("{}")
    with pytest.raises(CaseError, match=message):
        load_case(make_case(tmp_path, **kwargs))


@pytest.mark.parametrize("field_name", ["no_such_field", "office_use"])
def test_expected_must_name_writable_fields(tmp_path, field_name):
    cases = [load_case(make_case(tmp_path, expected={field_name: "x"}))]
    with pytest.raises(CaseError, match=field_name):
        validate_cases(cases, Settings())


def test_discover_cases(tmp_path):
    make_case(tmp_path, "b")
    make_case(tmp_path, "a")
    (tmp_path / "not_a_case").mkdir()
    assert [c.name for c in discover_cases(tmp_path)] == ["a", "b"]
    assert [c.name for c in discover_cases(tmp_path, ["b"])] == ["b"]
    with pytest.raises(CaseError, match="unknown case"):
        discover_cases(tmp_path, ["zzz"])
    with pytest.raises(CaseError, match="no cases"):
        discover_cases(tmp_path / "not_a_case")


def test_checked_in_cases_are_valid():
    cases = discover_cases()
    assert {c.name for c in cases} >= {
        "job_application",
        "w9_from_email",
        "intake_from_reference",
        "benefits_injection",
        "irs_w9",
        "irs_w4",
        "irs_8822",
        "irs_w8ben",
    }
    validate_cases(cases, Settings())
    for case in cases:
        assert case.today is not None and case.description
        assert any(case.expected.values()) and any(not v for v in case.expected.values())


def _oracle_mapper_for_case(case) -> FakeMapper:
    """Builds a FakeMapper whose proposals for `case.expected` are grounded in `request.sources`."""
    from pdf_wizard.guardrails import _ProposalChecker
    from pdf_wizard.models import Source, SourceKind

    def find_proposal(checker: _ProposalChecker, f: FormField, val: str, sources: list[Source]):
        ordered = [s for s in sources if s.kind is not SourceKind.SYSTEM] + [
            s for s in sources if s.kind is SourceKind.SYSTEM
        ]
        for s in ordered:
            cands: list[str] = []
            if s.kind is SourceKind.STRUCTURED and s.keypaths:
                keys = list(s.keypaths.keys())
                cands.extend(keys)
                for i, k1 in enumerate(keys):
                    for k2 in keys[i + 1 :]:
                        cands.append(f"{k1} + {k2}")
                    for k2 in keys[:i]:
                        cands.append(f"{k1} + {k2}")
                    for j, k2 in enumerate(keys[i + 1 :], i + 1):
                        for k3 in keys[j + 1 :]:
                            cands.append(f"{k1} + {k2} + {k3}")
            text = s.verification_text or s.text or ""
            lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= 4]
            cands.extend(lines)
            for width in (2, 3):
                for i in range(len(lines) - width + 1):
                    cands.append("\n".join(lines[i : i + width]))
            if not s.verifiable:
                cands.append(f"Scanned document states: {val}")
            for ev in cands:
                p = proposal(f.id, val, source_id=s.id, evidence=ev, confidence=0.99)
                chk = checker.check(f, p)
                if chk is not None and chk.accepted:
                    return p
        raise AssertionError(f"{case.name}: no passing evidence found for {f.name}={val!r}")

    def fn(req: MappingRequest):
        checker = _ProposalChecker(list(req.sources), 0.7, 0.9)
        out = []
        for f in req.fields:
            acc = case.expected.get(f.name, ())
            if not acc:
                continue
            for target in acc:
                if f.type is FieldType.CHECKBOX:
                    val = "checked"
                elif f.type in (FieldType.RADIO, FieldType.CHOICE):
                    val = target
                    for o in f.options:
                        if o.value.lower() == target.lower() or (o.label and o.label.lower() == target.lower()):
                            val = o.value
                            break
                else:
                    val = target
                try:
                    out.append(find_proposal(checker, f, val, list(req.sources)))
                    break
                except AssertionError:
                    continue
            else:
                raise AssertionError(f"{case.name}: could not find evidence for {f.name} -> {acc}")
        return out

    return FakeMapper(fn=fn, model="oracle")


@pytest.mark.parametrize("case", discover_cases(), ids=lambda c: c.name)
def test_checked_in_cases_fill_and_read_back_end_to_end(case):
    from pdf_wizard.pipeline import FillOptions, detect_form_schema, fill_form

    settings = Settings()
    mapper = _oracle_mapper_for_case(case)
    [run] = run_case(case, mapper, model="oracle", thresholds=[0.7], settings=settings)

    assert run.error is None
    m = run.metrics
    assert m.wrong == 0
    assert m.missed == 0
    assert m.hallucinated == 0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.hallucination_rate == 0.0

    # Also verify Stage 1 + Stage 2 split (`detect_form_schema` -> `fill_form(options=FillOptions(schema=...))`)
    form_bytes = case.form_path.read_bytes()
    schema = detect_form_schema(form_bytes, settings=settings, enrich=False)
    schema_mapper = _oracle_mapper_for_case(case)
    outcome = fill_form(
        form_bytes,
        case.inputs(),
        mapper=schema_mapper,
        settings=settings,
        options=FillOptions(schema=schema),
        form_name=f"{case.name}.pdf",
        today=case.today,
    )
    assert outcome.pdf is not None
    assert len(schema_mapper.requests) == 1
    assert schema_mapper.requests[0].form_pdf is None  # skipped redundant visual PDF upload
    assert len(outcome.raw_values) == m.correct


# --------------------------------------------------------------------------- #
# Runner CLI
# --------------------------------------------------------------------------- #


def run_cli(argv, factory, monkeypatch):
    monkeypatch.setattr(eval_run, "configure_logging", lambda level: None)  # leave global logging alone
    out, err = io.StringIO(), io.StringIO()
    code = eval_run.run([str(a) for a in argv], mapper_factory=factory, settings=Settings(), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_runner_prints_table_and_writes_results(tmp_path, monkeypatch):
    make_case(tmp_path / "cases")
    good = {k: v for k, v in PROPOSALS.items() if k not in ("email", "state")}
    out_file = tmp_path / "results" / "latest.json"
    code, out, err = run_cli(
        ["--cases", tmp_path / "cases", "--model", "m1", "--threshold", "0.6", "--threshold", "0.8",
         "--out", out_file],
        lambda model: name_mapper(good),
        monkeypatch,
    )
    assert code == 0, err
    assert "precision" in out and "Recommended: m1 at threshold 0.80" in out
    assert "missed: dob (no_data), ssn (no_data)" in out
    assert "Jane" not in out  # no values without --show-values
    data = json.loads(out_file.read_text())
    assert data["recommended"] == {"model": "m1", "threshold": 0.8}
    assert "[1/1] sample with m1" in err


def test_runner_show_values_strips_terminal_escapes(tmp_path, monkeypatch):
    make_case(tmp_path / "cases")
    mapper_spec = {**PROPOSALS, "full_name": ("Jane\x1b[31m Q. Public", "applicant.name")}
    code, out, _ = run_cli(["--cases", tmp_path / "cases", "--threshold", "0.7", "--show-values"],
                           lambda model: name_mapper(mapper_spec), monkeypatch)
    assert code == 1  # precision 50% (the name with an escape code is rejected) < 98%: no recommendation
    assert "No configuration reached the precision target" in out
    assert "wrong: state: got 'NY', expected 'TX'" in out
    assert "missed: dob (no_data): got (blank), expected '01/31/1990'" in out
    assert "\x1b" not in out


def test_runner_reports_case_errors(tmp_path, monkeypatch):
    make_case(tmp_path / "cases", expected={"bogus": "x"})
    code, _, err = run_cli(["--cases", tmp_path / "cases"], lambda model: name_mapper({}), monkeypatch)
    assert code == 2 and "bogus" in err


# --------------------------------------------------------------------------- #
# Synthetic dataset & fake PDF generator (`evals.dataset`)
# --------------------------------------------------------------------------- #


def test_dataset_generator_all_templates_and_formats(tmp_path):
    from evals.dataset import FORMATS, TEMPLATES, generate_dataset, main as dataset_main
    from pdf_wizard.pdf.inspector import open_form

    out_dir = tmp_path / "synth"
    buf = io.StringIO()
    rc = dataset_main(
        [
            "--template",
            "all",
            "--count",
            "6",
            "--formats",
            ",".join(FORMATS),
            "--seed",
            "123",
            "--out",
            str(out_dir),
        ],
        stdout=buf,
    )
    assert rc == 0
    assert "Generated 6 fake filled PDF(s)" in buf.getvalue()

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["count"] == 6
    assert set(manifest["templates"]) == set(TEMPLATES.keys())
    assert tuple(manifest["formats"]) == FORMATS

    # Verify every filled ground-truth PDF is readable and has populated widgets
    filled_files = sorted((out_dir / "filled").glob("*.pdf"))
    assert len(filled_files) == 6
    for pdf_path in filled_files:
        loaded = open_form(pdf_path.read_bytes(), limits=Settings().limits)
        populated = [f for f in loaded.info.fields if f.current_value not in (None, "", "/Off", "Off")]
        assert len(populated) >= 5, f"{pdf_path.name} should have populated fields"

    # Verify all generated cases validate and score 100% end-to-end
    cases = discover_cases(out_dir / "cases")
    assert len(cases) == 6
    validate_cases(cases, Settings())
    for case in cases:
        mapper = _oracle_mapper_for_case(case)
        [run] = run_case(case, mapper, model="oracle", thresholds=[0.7], settings=Settings())
        assert run.error is None
        assert run.metrics.precision == 1.0
        assert run.metrics.recall == 1.0
        assert run.metrics.hallucination_rate == 0.0

    # Also test custom --form (generic schema-driven Faker synthesizer)
    custom_out = tmp_path / "custom_synth"
    custom_manifest = generate_dataset(
        custom_out,
        form_path=TEMPLATES["job_application"],
        count=2,
        formats=("txt", "scanned_pdf"),
        seed=7,
    )
    assert len(custom_manifest.filled_pdfs) == 2
    custom_cases = discover_cases(custom_out / "cases")
    validate_cases(custom_cases, Settings())

