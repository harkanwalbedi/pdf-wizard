"""Synthetic personas for arbitrary forms (`python -m evals.dataset --form`).

A persona is a set of field values plus, for each value, the sentence of the
intake document that states it. Personas come from Gemini (a separate prompt,
and by default a separate model, from the filler) or from a small offline
Faker fallback. Either way, every persona goes through the product's own
guardrails before it becomes ground truth:

- physical checks: type, options, max_len, comb, format mask, one per group;
- grounding: the value must be stated by its sentence, and the sentence must
  be in the document the filler will read.

Rejected values are dropped from the document and from `expected.json`, so a
case never rewards or penalizes the filler for an ungroundable answer.
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from faker import Faker
from pydantic import BaseModel, Field

from pdf_wizard.config import Settings
from pdf_wizard.guardrails import apply_guardrails
from pdf_wizard.mapping.prompt import PromptPart, _catalog_line
from pdf_wizard.models import FieldProposal, FieldType, FormField, Source, SourceKind

GENERATOR_PROMPT_VERSION = "2026-10-08.1"
PERSONAS_PER_CALL = 5
_MAX_ROUNDS = 3  # validation rounds: dropping a sentence can orphan another value

GENERATOR_SYSTEM_PROMPT = """\
You write synthetic test data for a PDF form-filling system. You receive the
field catalog of one fillable form. Invent realistic, entirely fictional people
or organizations who would plausibly submit this form, and decide what each of
them would enter.

Rules:
- Use only fictional data. Use 555-01xx phone numbers, example.com/example.org
  e-mail domains, and identification numbers that are clearly synthetic.
- Be internally consistent: one coherent scenario per persona (dates in order,
  city/state/ZIP that agree, the same name everywhere it appears).
- Be realistic about blanks: leave office-use, signature, "if different", and
  branches that don't apply to the scenario empty. Not every field gets a value.
- Checkboxes: value "checked", and only for boxes that apply. At most one box
  per "group". Radio and choice fields: one of the listed option values, exactly.
- Text: respect max_len, comb (one character per box: digits only, no
  separators, exactly max_len characters), and the format mask if one is given.
- For every value, write `source_line`: one natural sentence, as it would appear
  in an intake e-mail or notes, that states the value verbatim (for checkboxes
  and options, the sentence must make the choice unambiguous using the words of
  the label or option). Do not mention field IDs, labels in quotes, or the form.
  Different values may not share a sentence.
- Vary the personas: different names, places, entity types, and branches.
"""


class PersonaValue(BaseModel):
    field_id: str = Field(description="Field ID from the catalog, e.g. 'F12'.")
    value: str = Field(description="Value to enter, formatted for the field.")
    source_line: str = Field(description="One sentence of the intake document stating this value verbatim.")


class Persona(BaseModel):
    scenario: str = Field(description="One line: who is filing and why.")
    values: list[PersonaValue]


class PersonaBatch(BaseModel):
    personas: list[Persona]


@dataclass(frozen=True)
class ValidatedPersona:
    pdf_values: dict[str, str]  # field id -> value for fill_pdf
    expected: dict[str, Any]  # field name -> accepted value(s), or None
    lines: list[str]  # the intake document, one sentence per kept value
    dropped: int  # values that failed a guardrail


def generator_model(settings: Settings, override: str | None) -> str:
    model = override or os.environ.get("PDF_WIZARD_GENERATOR_MODEL", "").strip() or settings.model
    if model == settings.model:
        print(
            f"Warning: the generator model ({model}) is also the filler model; synthetic cases will be biased "
            "toward it. Set --generator-model or PDF_WIZARD_GENERATOR_MODEL.",
            file=sys.stderr,
        )
    return model


def generate_with_gemini(
    fields: Sequence[FormField], count: int, *, mapper: Any, model: str, seed: int
) -> list[Persona]:
    """Asks Gemini for `count` personas, PERSONAS_PER_CALL at a time."""
    catalog = "\n".join(_catalog_line(f) for f in fields if f.writable)
    personas: list[Persona] = []
    while len(personas) < count:
        n = min(PERSONAS_PER_CALL, count - len(personas))
        avoid = "\n".join(f"- {p.scenario}" for p in personas[-20:]) or "(none yet)"
        text = (
            f"FIELD CATALOG (one JSON object per line):\n{catalog}\n\n"
            f"Write {n} personas (variation seed {seed}-{len(personas)}). "
            f"Scenarios already used, do not repeat:\n{avoid}\n"
        )
        batch: PersonaBatch = mapper.generate(
            [PromptPart(text=text)], system_prompt=GENERATOR_SYSTEM_PROMPT, response_schema=PersonaBatch, model=model
        )
        if not batch.personas:
            raise RuntimeError("the generator model returned no personas")
        personas.extend(batch.personas[:n])
    return personas


def generate_with_faker(fields: Sequence[FormField], fake: Faker, rng: random.Random) -> Persona:
    """Offline fallback: values chosen from the field's type and constraints only."""
    values: list[PersonaValue] = []
    used_groups: set[str] = set()
    for f in fields:
        if not f.writable or f.office_use_only:
            continue
        label = (f.label or f.name).rstrip(" :.")
        if f.type is FieldType.CHECKBOX:
            if not f.group or f.group in used_groups:
                continue
            used_groups.add(f.group)
            values.append(PersonaValue(field_id=f.id, value="checked", source_line=f"Selected: {label}."))
            continue
        if f.type in (FieldType.RADIO, FieldType.CHOICE):
            opts = [o for o in f.options if o.value.strip() and o.value != "Off"]
            if opts:
                o = rng.choice(opts)
                values.append(PersonaValue(field_id=f.id, value=o.value, source_line=f"{label}: {o.label or o.value}."))
            continue
        hint = f.format_hint or ""
        if f.comb and f.max_len:
            value = "".join(str(rng.randint(1, 9)) for _ in range(f.max_len))
        elif hint.startswith("date:"):
            value = fake.date_between("-5y", "today").isoformat()
        elif hint.startswith(("number:", "percent:")):
            value = str(rng.randint(1, 250))
        else:
            value = fake.name() if f.max_len is None or f.max_len >= 20 else fake.lexify("?" * min(f.max_len, 6)).upper()
            value = value[: f.max_len] if f.max_len else value
        values.append(PersonaValue(field_id=f.id, value=value, source_line=f"{label}: {value}."))
    return Persona(scenario="faker fallback", values=values)


def validate_persona(fields: Sequence[FormField], persona: Persona) -> ValidatedPersona:
    """Runs the persona through the product guardrails until every kept value is grounded."""
    by_id = {f.id: f for f in fields}
    items = [v for v in persona.values if v.field_id in by_id and v.value.strip() and v.source_line.strip()]
    dropped = len(persona.values) - len(items)
    outcome = None
    for _ in range(_MAX_ROUNDS):
        lines = list(dict.fromkeys(_one_line(v.source_line) for v in items))
        source = Source(id="S1", kind=SourceKind.TEXT, name="intake", text="\n".join(lines))
        proposals = [
            FieldProposal(
                field_id=v.field_id, value=v.value, source_id="S1", evidence=_one_line(v.source_line),
                confidence=1.0, sensitive=False,
            )
            for v in items
        ]
        outcome = apply_guardrails(list(fields), [source], proposals, min_confidence=0.0, min_confidence_unverifiable=0.0)
        kept = [v for v in items if v.field_id in outcome.values]
        if len(kept) == len(items):
            break
        dropped += len(items) - len(kept)
        items = kept
    assert outcome is not None
    accepted = {v.field_id: v for v in items if v.field_id in outcome.values}
    expected: dict[str, Any] = {}
    for f in fields:
        if not f.writable:
            continue
        pdf_value = outcome.values.get(f.id)
        if f.id not in accepted or pdf_value is None or pdf_value == "Off":
            expected[f.name] = None
        elif f.type is FieldType.CHECKBOX:
            expected[f.name] = True
        else:
            raw = accepted[f.id].value
            expected[f.name] = pdf_value if raw == pdf_value else [pdf_value, raw]
    pdf_values = {fid: val for fid, val in outcome.values.items() if fid in accepted and val != "Off"}
    lines = list(dict.fromkeys(_one_line(accepted[fid].source_line) for fid in pdf_values))
    return ValidatedPersona(pdf_values=pdf_values, expected=expected, lines=lines, dropped=dropped)


def persona_record(persona: Persona, *, generator: str, model: str | None) -> str:
    """The raw persona as saved to `cases/<id>/persona.json` (synthetic data only)."""
    data = {"generator": generator, "model": model, "prompt_version": GENERATOR_PROMPT_VERSION, **persona.model_dump()}
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def load_persona(text: str) -> Persona:
    return Persona.model_validate(json.loads(text))


def _one_line(text: str) -> str:
    return " ".join(text.split())
