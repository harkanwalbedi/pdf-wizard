"""Code-based checks on model proposals (design doc, Section 6).

Principle: a blank field is better than a wrong one. A proposal is accepted only
if it passes every check below; anything else leaves the field blank and is
explained in the report. Details never contain values.

Checks, in order: field exists and is writable, source exists, confidence is a
number in [0, 1], value is not empty, type/option/length/format checks, evidence
grounding (the quote or key path is in the source), an instruction-pattern
check, value support (the evidence states the value; see evidence.py),
confidence threshold.
"""

from __future__ import annotations

import calendar
import math
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache

from .evidence import value_supported
from .models import (
    FieldProposal,
    FieldResult,
    FieldStatus,
    FieldType,
    FormField,
    Option,
    RejectReason,
    Source,
    SourceKind,
)
from .privacy import is_sensitive

_MAX_HINT_PATTERN_CHARS = 64  # longer date/mask patterns from the PDF are ignored
# Evidence is a short quote; longer text is rejected before the regex checks run on it.
_MAX_EVIDENCE_CHARS = 2_000
_MIN_EVIDENCE_CHARS = 4  # shorter quotes ("e", "=") occur almost anywhere
_KEYPATH_JOIN = re.compile(r"\s+\+\s+")  # "applicant.first + applicant.last"

# Characters that may be dropped to fit a comb field or MaxLen ("123-45-6789" -> "123456789").
_SEPARATORS = re.compile(r"[-/ .()]")
_PLAIN_DECIMAL = re.compile(r"[+-]?\d*\.\d+")  # never drop the point from "12.50"

_CHECKED_WORDS = frozenset({"checked", "yes", "true", "on", "x"})
_UNCHECKED_WORDS = frozenset({"unchecked", "no", "false", "off"})

# A model that follows an instruction injected into a source can cite the
# injected text verbatim, which passes the substring check. Evidence must be a
# fact, so instruction-like evidence is rejected (defense in depth; user notes
# are exempt because they are the user's own instructions).
_INSTRUCTION_PATTERNS = tuple(
    re.compile(p)
    for p in (
        r"\bignore (?:all |any )?(?:the )?(?:previous|prior|above|earlier) (?:instructions|rules|prompts)\b",
        r"\bdisregard (?:the |all )?(?:previous|prior|above)\b",
        r"\bsystem prompt\b",
        r"\byou are (?:now )?an? (?:ai|assistant|model)\b",
        r"\b(?:put|enter|write|fill|use) .{0,40} (?:in|into) (?:the |this )?.{0,30}field\b",
        r"\bnew instructions\b",
    )
)

# Evidence normalization: curly quotes -> straight, dashes -> "-".
_PUNCTUATION = str.maketrans(
    {
        **dict.fromkeys("\u2018\u2019\u201a\u201b\u2032", "'"),
        **dict.fromkeys("\u201c\u201d\u201e\u201f\u2033", '"'),
        **dict.fromkeys("\u2010\u2011\u2012\u2013\u2014\u2015\u2212", "-"),
    }
)

_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)  # fmt: skip
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

# Acrobat date tokens (AFDate_FormatEx / util.printd), longest first per letter.
# Each maps to (regex group name, regex).
_DATE_TOKENS: dict[str, tuple[str, str]] = {
    "yyyy": ("year4", r"[0-9]{4}"),
    "yy": ("year2", r"[0-9]{2}"),
    "mmmm": ("month_name", "|".join(_MONTHS)),
    "mmm": ("month_abbr", "|".join(m[:3] for m in _MONTHS)),
    "mm": ("month", r"[0-9]{2}"),
    "m": ("month", r"[0-9]{1,2}"),
    "dddd": ("weekday", "|".join(_WEEKDAYS)),
    "ddd": ("weekday_abbr", "|".join(w[:3] for w in _WEEKDAYS)),
    "dd": ("day", r"[0-9]{2}"),
    "d": ("day", r"[0-9]{1,2}"),
    "HH": ("hour24", r"[0-9]{2}"),
    "H": ("hour24", r"[0-9]{1,2}"),
    "hh": ("hour12", r"[0-9]{2}"),
    "h": ("hour12", r"[0-9]{1,2}"),
    "MM": ("minute", r"[0-9]{2}"),
    "M": ("minute", r"[0-9]{1,2}"),
    "ss": ("second", r"[0-9]{2}"),
    "s": ("second", r"[0-9]{1,2}"),
    "tt": ("ampm", "am|pm"),
    "t": ("ampm1", "a|p"),
}
_TIME_TOKENS = frozenset({"HH", "H", "hh", "h", "MM", "M", "ss", "s", "tt", "t"})


@dataclass
class GuardrailOutcome:
    values: dict[str, str] = field(repr=False)  # field id -> PDF-ready value (see pdf.filler.fill_pdf)
    results: dict[str, FieldResult] = field(default_factory=dict)  # every field, in form order
    warnings: list[str] = field(default_factory=list)


def apply_guardrails(
    fields: list[FormField],
    sources: list[Source],
    proposals: list[FieldProposal],
    *,
    min_confidence: float,
    min_confidence_unverifiable: float,
) -> GuardrailOutcome:
    """Checks every proposal and decides the outcome of every field.

    `values` holds only accepted values; `results` explains every field.
    """
    fields_by_id = {f.id: f for f in fields}
    checker = _ProposalChecker(sources, min_confidence, min_confidence_unverifiable)
    checked: dict[str, list[_Checked]] = {}
    proposed_but_not_writable: set[str] = set()
    unknown_fields = 0

    for proposal in proposals:
        field_id = proposal.field_id.strip()
        target = fields_by_id.get(field_id)
        if target is None:
            unknown_fields += 1
        elif not target.writable:
            proposed_but_not_writable.add(field_id)
        elif (outcome := checker.check(target, proposal)) is not None:
            checked.setdefault(field_id, []).append(outcome)

    values: dict[str, str] = {}
    results: dict[str, FieldResult] = {}
    checked_by_group: dict[str, list[str]] = {}
    for f in fields:
        result, value = _decide(f, checked.get(f.id, []), f.id in proposed_but_not_writable)
        results[f.id] = result
        if value is not None:
            values[f.id] = value
            if f.type is FieldType.CHECKBOX and f.group and result.value == "checked":
                checked_by_group.setdefault(f.group, []).append(f.id)

    for fids in checked_by_group.values():
        if len(fids) > 1:
            for fid in fids:
                values.pop(fid, None)
                results[fid] = results[fid].model_copy(
                    update={
                        "status": FieldStatus.REJECTED,
                        "reason": RejectReason.INVALID_VALUE,
                        "detail": "Only one checkbox in this group may be checked",
                    }
                )

    warnings = []
    if unknown_fields == 1:
        warnings.append("1 proposal referred to an unknown field and was ignored.")
    elif unknown_fields:
        warnings.append(f"{unknown_fields} proposals referred to unknown fields and were ignored.")
    return GuardrailOutcome(values=values, results=results, warnings=warnings)


# --------------------------------------------------------------------------- #
# Per-proposal checks
# --------------------------------------------------------------------------- #


class _Invalid(Exception):
    """A value failed a type, option, length, or format check. `detail` is safe to show."""

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


@dataclass
class _Checked:
    proposal: FieldProposal
    source_id: str | None
    pdf_value: str | None = None  # accepted: value written to the PDF
    display: str | None = None  # accepted: value shown in the report
    reason: RejectReason | None = None
    detail: str | None = None

    @property
    def accepted(self) -> bool:
        return self.reason is None


class _ProposalChecker:
    def __init__(self, sources: list[Source], min_confidence: float, min_confidence_unverifiable: float):
        self._sources = {s.id: s for s in sources}
        self._min_confidence = min_confidence
        self._min_confidence_unverifiable = min_confidence_unverifiable
        self._normalized: dict[str, str] = {}  # source id -> normalized verification text
        self._keypath_values: dict[str, set[str]] = {}  # source id -> normalized structured values

    def check(self, f: FormField, p: FieldProposal) -> _Checked | None:
        """Returns the checked proposal, or None if it should be ignored (empty value)."""
        source = self._sources.get(p.source_id.strip())
        if source is None:
            # Keep the model's source ID only if it is shaped like one.
            shown = p.source_id.strip() if re.fullmatch(r"S\d{1,4}", p.source_id.strip()) else None
            return _Checked(p, shown, reason=RejectReason.UNKNOWN_SOURCE, detail="Cites a source that doesn't exist")
        reject = _Rejector(p, source.id)
        if not _valid_confidence(p.confidence):
            return reject(RejectReason.INVALID_VALUE, "Confidence is not a number between 0 and 1")
        if not p.value.strip():
            return None
        if f.office_use_only:
            return reject(RejectReason.INVALID_VALUE, "This field is for office use only")
        try:
            pdf_value, display = _check_value(f, p.value)
        except _Invalid as exc:
            return reject(RejectReason.INVALID_VALUE, exc.detail)
        evidence = p.evidence.strip()
        if not evidence:
            return reject(RejectReason.UNVERIFIED, "No evidence was given")
        if len(evidence) > _MAX_EVIDENCE_CHARS:
            return reject(RejectReason.UNVERIFIED, "Evidence is too long to check")
        support = self._support(source, evidence)
        if support is None:
            return reject(RejectReason.UNVERIFIED, f"Evidence not found in {source.id}")
        if support.quote and _too_short(f, p.value, evidence):
            return reject(RejectReason.UNVERIFIED, "Evidence is too short to check")
        if source.kind is not SourceKind.NOTE and _looks_like_instruction(evidence):
            return reject(RejectReason.UNVERIFIED, "The evidence looks like an instruction, not a fact.")
        if not value_supported(f, p.value, pdf_value, values=support.values, context=support.context):
            return reject(RejectReason.UNVERIFIED, "The evidence doesn't state this value")
        threshold = self._min_confidence if source.verifiable else self._min_confidence_unverifiable
        if p.confidence < threshold:
            detail = f"Confidence {_pct(p.confidence, threshold)} is below {_pct(threshold, p.confidence)}"
            if not source.verifiable:
                detail += " (required for sources that can't be checked automatically)"
            return reject(RejectReason.LOW_CONFIDENCE, detail)
        return _Checked(p, source.id, pdf_value=pdf_value, display=display)

    def _support(self, source: Source, evidence: str) -> _Support | None:
        """What the evidence stands for, or None if it isn't in the source.

        Structured sources: key paths (several joined by " + "), a "path = value"
        line, or a quote of a whole stored value; the support is the stored
        values, never the model's copy of them. Other sources: the quote, once
        found. Sources without checkable text (images, scans): the quote, unchecked.
        """
        if source.kind is SourceKind.STRUCTURED and source.keypaths:
            return self._keypath_support(source, evidence)
        if source.verifiable and not self._quote_found(source, evidence):
            return None
        return _Support(values=evidence, context=evidence, quote=True)

    def _keypath_support(self, source: Source, evidence: str) -> _Support | None:
        if source.id not in self._keypath_values:
            self._keypath_values[source.id] = {_normalize_evidence(v) for v in source.keypaths.values()}
        values: list[str] = []
        context: list[str] = []
        for part in [evidence] if evidence in source.keypaths else _KEYPATH_JOIN.split(evidence):
            part = part.strip()
            key = part if part in source.keypaths else part.split(" = ", 1)[0].strip()
            if key in source.keypaths:
                values.append(source.keypaths[key])
                context += [key, source.keypaths[key]]
            elif part and _normalize_evidence(part) in self._keypath_values[source.id]:
                values.append(part)
                context.append(part)
            else:
                return None
        return _Support(values="\n".join(values), context="\n".join(context), quote=False)

    def _quote_found(self, source: Source, evidence: str) -> bool:
        needle = _normalize_evidence(evidence)
        if source.id not in self._normalized:
            self._normalized[source.id] = _normalize_evidence(source.verification_text)
        haystack = self._normalized[source.id]
        if needle in haystack:
            return True
        # Allow the model's ellipses ("..." or "…", which NFKC turns into "..."):
        # every non-empty part must appear, in order, and say something.
        if "..." not in needle:
            return False
        parts = [part.strip(" .") for part in needle.split("...")]
        parts = [part for part in parts if part]
        if not parts or not all(any(ch.isalnum() for ch in part) for part in parts):
            return False
        position = 0
        for part in parts:
            found = haystack.find(part, position)
            if found < 0:
                return False
            position = found + len(part)
        return True


@dataclass(frozen=True)
class _Support:
    values: str  # what the evidence states; a text value must appear in it
    context: str  # values plus structured key paths, which may name an option
    quote: bool  # the model's own words (subject to the minimum length)


def _too_short(f: FormField, value: str, evidence: str) -> bool:
    """A quote needs at least 4 characters, or as many as the value if it is
    shorter. Checkboxes always need 4: the value doesn't bind them."""
    have = len("".join(_normalize_evidence(evidence).split()))
    need = _MIN_EVIDENCE_CHARS
    if f.type is not FieldType.CHECKBOX:
        need = min(need, len("".join(_normalize_evidence(value).split())))
    return have < need


class _Rejector:
    def __init__(self, proposal: FieldProposal, source_id: str):
        self._proposal, self._source_id = proposal, source_id

    def __call__(self, reason: RejectReason, detail: str) -> _Checked:
        return _Checked(self._proposal, self._source_id, reason=reason, detail=detail)


def _normalize_evidence(text: str) -> str:
    """NFKC, casefold, straight quotes, ASCII dashes, collapsed whitespace."""
    text = unicodedata.normalize("NFKC", text).casefold().translate(_PUNCTUATION)
    return " ".join(text.split())


def _looks_like_instruction(evidence: str) -> bool:
    normalized = _normalize_evidence(evidence)
    return any(pattern.search(normalized) for pattern in _INSTRUCTION_PATTERNS)


def _valid_confidence(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def _pct(value: float, other: float) -> str:
    """Formats a confidence with 2 decimals, or 3 if 2 would hide a difference."""
    text = f"{value:.2f}"
    return f"{value:.3f}" if text == f"{other:.2f}" and value != other else text


# --------------------------------------------------------------------------- #
# Deciding each field
# --------------------------------------------------------------------------- #


_NOT_WRITABLE_NOTE = "A value was proposed for this field, but it can't be filled, so it was ignored."


def _decide(f: FormField, checked: list[_Checked], proposed_but_not_writable: bool) -> tuple[FieldResult, str | None]:
    """Returns the field's result and its PDF value (None unless filled)."""
    base = {"field_id": f.id, "name": f.name, "label": f.label, "type": f.type, "page": f.page}
    if not f.writable:
        notes = [_NOT_WRITABLE_NOTE] if proposed_but_not_writable else []
        return FieldResult(**base, status=FieldStatus.NOT_WRITABLE, notes=notes), None
    accepted = [c for c in checked if c.accepted]
    if accepted:
        return _filled(f, base, accepted)
    if checked:
        return _rejected(f, base, checked), None
    return FieldResult(**base, status=FieldStatus.NO_DATA), None


def _filled(f: FormField, base: dict, accepted: list[_Checked]) -> tuple[FieldResult, str | None]:
    best = max(accepted, key=lambda c: c.proposal.confidence)  # first one wins ties
    notes = []
    if len({c.pdf_value for c in accepted}) > 1:
        source_ids = ", ".join(dict.fromkeys(c.source_id for c in accepted if c.source_id))
        notes.append(f"Conflicting values from {source_ids}; kept the highest-confidence value.")
    p = best.proposal
    result = FieldResult(
        **base,
        status=FieldStatus.FILLED,
        value=best.display,
        source_id=best.source_id,
        evidence=p.evidence,
        confidence=p.confidence,
        sensitive=p.sensitive or is_sensitive(f, p.value) or is_sensitive(f, best.display or ""),
        notes=notes,
    )
    return result, best.pdf_value


def _rejected(f: FormField, base: dict, checked: list[_Checked]) -> FieldResult:
    """Reports the highest-confidence rejected proposal (the report masks its value later)."""
    best = max(checked, key=lambda c: c.proposal.confidence if _valid_confidence(c.proposal.confidence) else -1.0)
    p = best.proposal
    others = len(checked) - 1
    notes = [f"{others} other proposal{'s' if others > 1 else ''} for this field also failed the checks."]
    finite = isinstance(p.confidence, (int, float)) and math.isfinite(p.confidence)
    return FieldResult(
        **base,
        status=FieldStatus.REJECTED,
        value=p.value,
        source_id=best.source_id,
        evidence=p.evidence,
        confidence=p.confidence if finite else None,
        sensitive=p.sensitive or is_sensitive(f, p.value),
        reason=best.reason,
        detail=best.detail,
        notes=notes if others else [],
    )


# --------------------------------------------------------------------------- #
# Value checks by field type
# --------------------------------------------------------------------------- #


def _check_value(f: FormField, raw: str) -> tuple[str, str]:
    """Validates and normalizes a proposed value for `f`.

    Returns (PDF-ready value, display value). Raises _Invalid with a safe detail.
    """
    if f.type is FieldType.TEXT:
        text = _check_text(f, raw)
        return text, text
    if f.type is FieldType.CHECKBOX:
        return _check_checkbox(f, raw)
    if f.type is FieldType.RADIO:
        value = _check_radio(f, raw)
        return value, value
    if f.type is FieldType.CHOICE:
        value = _check_choice(f, raw)
        return value, value
    raise _Invalid("This type of field can't be filled")


def _check_text(f: FormField, raw: str) -> str:
    value = raw.replace("\r\n", "\n").replace("\r", "\n").strip()
    if _has_forbidden_chars(value):
        raise _Invalid("Contains control or invisible characters")
    if not f.multiline:
        value = " ".join(value.split())
    value, numeric = _apply_format_hint(f.format_hint, value)
    return _fit_length(f, value, may_drop_separators=not numeric)


def _has_forbidden_chars(value: str) -> bool:
    for i, ch in enumerate(value):
        if ch in _JOINERS and 0 < i < len(value) - 1 and _joinable(value[i - 1]) and _joinable(value[i + 1]):
            continue
        if _forbidden_char(ch):
            return True
    return False


_JOINERS = "\u200c\u200d"  # ZWNJ and ZWJ: part of the spelling of Persian and Indic names


def _joinable(ch: str) -> bool:
    return not ch.isascii() and (ch.isalpha() or unicodedata.category(ch).startswith("M"))


def _forbidden_char(ch: str) -> bool:
    """Controls (except newline and tab), invisible format characters such as
    zero-width spaces (they also defeat the sensitive-value patterns),
    surrogates, private-use, unassigned code points, and the Unicode line and
    paragraph separators (only \\n may break a line)."""
    category = unicodedata.category(ch)
    return (category == "Cc" and ch not in "\n\t") or category in ("Cf", "Cs", "Co", "Cn", "Zl", "Zp")


def _fit_length(f: FormField, value: str, *, may_drop_separators: bool) -> str:
    """Enforces MaxLen without truncating; comb fields get digits only when possible."""
    max_len = f.max_len if f.max_len and f.max_len > 0 else None
    if max_len is None:
        return value
    too_long = len(value) > max_len
    if (f.comb or too_long) and may_drop_separators and not _PLAIN_DECIMAL.fullmatch(value):
        digits = _SEPARATORS.sub("", value)
        if digits and digits.isascii() and digits.isdigit() and len(digits) <= max_len:
            return digits
    if too_long:
        raise _Invalid(f"Longer than the {max_len}-character limit")
    return value


def _check_checkbox(f: FormField, raw: str) -> tuple[str, str]:
    key = _option_key(raw)
    options = [o for o in f.options if o.value != "Off"]
    on_states = list(dict.fromkeys(o.value for o in options))
    named = _match_options(options, key)
    if key in _UNCHECKED_WORDS:
        if named:
            raise _Invalid("Ambiguous: the value names an on-state of the checkbox but also means unchecked")
        return "Off", "unchecked"
    if len(named) == 1:
        return named[0], "checked"
    if len(named) > 1:
        raise _Invalid("Matches more than one state of the checkbox")
    if key in _CHECKED_WORDS:
        if len(on_states) == 1:
            return on_states[0], "checked"
        if not on_states:
            raise _Invalid("The checkbox has no known on-state")
        raise _Invalid("The checkbox has several on-states; the value must name one")
    raise _Invalid("Not a checkbox value (expected checked or unchecked)")


def _check_radio(f: FormField, raw: str) -> str:
    key = _option_key(raw)
    if key in ("off", "unchecked"):
        raise _Invalid("Radio buttons can't be cleared; the value must be one of the options")
    matches = _match_options([o for o in f.options if o.value != "Off"], key)
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise _Invalid("Matches more than one option")
    raise _Invalid("Not one of the allowed options")


def _check_choice(f: FormField, raw: str) -> str:
    matches = _match_options(f.options, _option_key(raw))
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise _Invalid("Matches more than one option")
    if f.editable:  # combo box that accepts free text; MaxLen still applies
        return _check_text(f, raw)
    raise _Invalid("Not one of the allowed options")


def _option_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def _match_options(options: list[Option], key: str) -> list[str]:
    """Option values whose value matches `key`; failing that, whose label matches."""
    by_value = [o.value for o in options if _option_key(o.value) == key]
    if by_value:
        return list(dict.fromkeys(by_value))
    return list(dict.fromkeys(o.value for o in options if o.label and _option_key(o.label) == key))


# --------------------------------------------------------------------------- #
# Format hints (parsed from Acrobat format scripts by the inspector)
# --------------------------------------------------------------------------- #


def _apply_format_hint(hint: str | None, value: str) -> tuple[str, bool]:
    """Returns (formatted value, is_numeric). Unknown hints are not checked."""
    kind, _, arg = (hint or "").strip().partition(":")
    kind = kind.lower()
    if kind == "date":
        return _check_date(value, arg.strip()), False
    if kind == "zip":
        if not re.fullmatch(r"[0-9]{5}", value):
            raise _Invalid("Not a 5-digit ZIP code")
        return value, False
    if kind == "zip4":
        digits = _digits_only(value, allowed=r"[0-9\s\-]+")
        if digits is None or len(digits) != 9:
            raise _Invalid("Not a ZIP+4 code (9 digits)")
        return f"{digits[:5]}-{digits[5:]}", False
    if kind == "phone":
        return _check_phone(value), False
    if kind == "ssn":
        digits = _digits_only(value, allowed=r"[0-9\s\-.]+")
        if digits is None or len(digits) != 9:
            raise _Invalid("Not a 9-digit SSN")
        return f"{digits[:3]}-{digits[3:5]}-{digits[5:]}", False
    if kind == "number":
        return _check_number(value, arg, percent=False), True
    if kind == "percent":
        return _check_number(value, arg, percent=True), True
    if kind == "mask":
        return _check_mask(value, arg), False
    return value, False


def _digits_only(value: str, *, allowed: str) -> str | None:
    if not re.fullmatch(allowed, value):
        return None
    return re.sub(r"[^0-9]", "", value)


def _check_phone(value: str) -> str:
    """US numbers: 10 digits (after an optional +1/1 prefix) or 7 digits."""
    international = value.startswith("+")
    digits = _digits_only(value, allowed=r"\+?[0-9\s\-.()/]+")
    if digits is not None and len(digits) == 11 and digits.startswith("1"):
        digits, international = digits[1:], False
    if digits is not None and not international:
        if len(digits) == 10:
            return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        if len(digits) == 7:
            return f"{digits[:3]}-{digits[3:]}"
    raise _Invalid("Not a 10-digit or 7-digit phone number")


def _check_number(value: str, arg: str, *, percent: bool) -> str:
    """Number fields store a plain number; percent fields store the fraction (25% -> 0.25)."""
    sep_style = _sep_style(arg)
    text = value
    if percent:
        if text.count("%") != 1:
            raise _Invalid("Not a percentage (expected a number with a % sign)")
        text = text.replace("%", "")
    # Drop currency symbols and spaces.
    text = "".join(ch for ch in text if not ch.isspace() and unicodedata.category(ch) != "Sc")
    negative = False
    if len(text) >= 2 and text[0] == "(" and text[-1] == ")":  # accounting style
        negative, text = True, text[1:-1]
    if text[:1] in ("-", "+", "\u2212"):
        negative, text = negative or text[0] != "+", text[1:]
    comma_decimal = sep_style in (2, 3)
    thousands, decimal = (r"\.", ",") if comma_decimal else (",", r"\.")
    match = re.fullmatch(
        rf"(?P<int>[0-9]{{1,3}}(?:{thousands}[0-9]{{3}})+|[0-9]*)(?:{decimal}(?P<frac>[0-9]*))?", text
    )
    if not match or not (match["int"] or match["frac"]):
        raise _Invalid("Not a number in the expected format")
    integer = re.sub(r"[^0-9]", "", match["int"]) or "0"
    fraction = match["frac"] or ""
    sign = "-" if negative else ""
    if percent:
        return sign + _shift_two_places(integer, fraction)
    point = "," if comma_decimal else "."
    return sign + integer + (point + fraction if fraction else "")


def _sep_style(arg: str) -> int:
    """Separator style from "<decimals>:<sep style>"; 0 (1,234.56) if missing."""
    parts = arg.split(":")
    try:
        return int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        return 0


def _shift_two_places(integer: str, fraction: str) -> str:
    """Exact decimal division by 100 on digit strings: ("25", "") -> "0.25"."""
    digits = integer + fraction
    point = len(integer) - 2
    if point <= 0:
        whole, frac = "0", "0" * -point + digits
    else:
        whole, frac = digits[:point], digits[point:]
    whole, frac = whole.lstrip("0") or "0", frac.rstrip("0")
    return whole + ("." + frac if frac else "")


def _check_mask(value: str, mask: str) -> str:
    """AFSpecial_KeystrokeEx masks: 9 digit, A letter, O letter or digit, X any character."""
    if not mask or len(mask) > _MAX_HINT_PATTERN_CHARS:
        return value
    if not _mask_regex(mask).fullmatch(value):
        raise _Invalid(f"Does not match the pattern {mask}")
    return value


@lru_cache(maxsize=256)
def _mask_regex(mask: str) -> re.Pattern[str]:
    classes = {"9": "[0-9]", "A": "[A-Za-z]", "O": "[A-Za-z0-9]", "X": "."}
    return re.compile("".join(classes.get(ch, re.escape(ch)) for ch in mask))


def _check_date(value: str, fmt: str) -> str:
    if not fmt or len(fmt) > _MAX_HINT_PATTERN_CHARS:
        return value
    tokens, pattern = _compile_date_format(fmt)
    match = pattern.fullmatch(value)
    if match:
        if _date_parts_valid(match):
            return value
        raise _Invalid(f"Not a valid date for the format {fmt}")
    has_time = any(is_token and text in _TIME_TOKENS for is_token, text in tokens)
    has_date = any(is_token and text not in _TIME_TOKENS for is_token, text in tokens)
    iso = _parse_iso_date(value)
    if iso is not None and has_date and not has_time:
        return _render_date(tokens, iso)  # deterministic reformat of YYYY-MM-DD
    raise _Invalid(f"Does not match the date format {fmt}")


def _tokenize_date_format(fmt: str) -> tuple[tuple[bool, str], ...]:
    tokens: list[tuple[bool, str]] = []
    i = 0
    while i < len(fmt):
        token = next((t for t in _DATE_TOKENS if fmt.startswith(t, i)), None)
        if token is None:
            tokens.append((False, fmt[i]))
            i += 1
        else:
            tokens.append((True, token))
            i += len(token)
    return tuple(tokens)


@lru_cache(maxsize=256)
def _compile_date_format(fmt: str) -> tuple[tuple[tuple[bool, str], ...], re.Pattern[str]]:
    tokens = _tokenize_date_format(fmt)
    seen: set[str] = set()
    parts = []
    for is_token, text in tokens:
        if not is_token:
            parts.append(re.escape(text))
            continue
        name, regex = _DATE_TOKENS[text]
        parts.append(f"(?P={name})" if name in seen else f"(?P<{name}>{regex})")
        seen.add(name)
    # ASCII-only case folding: with Unicode folding, "ı" (dotless i) or "ſ" (long s) would
    # match month and weekday names but fail the name lookup in _date_parts_valid.
    return tokens, re.compile("".join(parts), re.IGNORECASE | re.ASCII)


def _date_parts_valid(match: re.Match[str]) -> bool:
    g = {k: v for k, v in match.groupdict().items() if v is not None}
    months = set()
    if "month" in g:
        months.add(int(g["month"]))
    for key, names in (("month_name", _MONTHS), ("month_abbr", [m[:3] for m in _MONTHS])):
        if key in g:
            position = _name_index(names, g[key])
            if position is None:
                return False
            months.add(position + 1)
    if len(months) > 1:
        return False
    month = months.pop() if months else None
    day = int(g["day"]) if "day" in g else None
    if (month is not None and not 1 <= month <= 12) or (day is not None and not 1 <= day <= 31):
        return False
    weekday = None
    for key, names in (("weekday", _WEEKDAYS), ("weekday_abbr", [w[:3] for w in _WEEKDAYS])):
        if key in g:
            weekday = _name_index(names, g[key])
            if weekday is None:
                return False
    if "year4" in g:
        years = [int(g["year4"])]
    elif "year2" in g:
        years = [1900 + int(g["year2"]), 2000 + int(g["year2"])]
    else:
        years = []
    if month is not None and day is not None:
        if years and not any(_real_date(y, month, day, weekday) for y in years):
            return False
        if not years and day > calendar.monthrange(2000, month)[1]:  # 2000 is a leap year
            return False
    limits = {"hour24": (0, 23), "hour12": (1, 12), "minute": (0, 59), "second": (0, 59)}
    return all(lo <= int(g[key]) <= hi for key, (lo, hi) in limits.items() if key in g)


def _name_index(names: list[str] | tuple[str, ...], text: str) -> int | None:
    """Position of `text` in `names`, ignoring ASCII case; None if absent."""
    wanted = text.lower() if text.isascii() else None
    return next((i for i, name in enumerate(names) if name.lower() == wanted), None)


def _real_date(year: int, month: int, day: int, weekday: int | None) -> bool:
    try:
        d = date(year, month, day)
    except ValueError:
        return False
    return weekday is None or d.weekday() == weekday


def _parse_iso_date(value: str) -> date | None:
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _render_date(tokens: tuple[tuple[bool, str], ...], d: date) -> str:
    render = {
        "yyyy": f"{d.year:04d}",
        "yy": f"{d.year % 100:02d}",
        "mmmm": _MONTHS[d.month - 1],
        "mmm": _MONTHS[d.month - 1][:3],
        "mm": f"{d.month:02d}",
        "m": str(d.month),
        "dddd": _WEEKDAYS[d.weekday()],
        "ddd": _WEEKDAYS[d.weekday()][:3],
        "dd": f"{d.day:02d}",
        "d": str(d.day),
    }
    return "".join(render[text] if is_token else text for is_token, text in tokens)
