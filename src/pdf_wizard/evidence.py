"""Value support: the evidence must state the proposed value (design doc, Section 6).

Grounding (guardrails.py) proves that the evidence occurs in the cited source.
This module proves the other half: that the evidence states the value. Only the
changes the prompt allows are tolerated: case, spacing, punctuation and
separators, accents on Latin letters, date formats, reordered or dropped words
("Doe, Jane" -> "Jane"), initials, and common US address abbreviations
("Street" -> "St", "Texas" -> "TX"). Without this check, a model that follows an
instruction injected into a source could cite any real quote and fill any value.

Checkboxes and yes/no answers judge the evidence instead of copying it, so they
are exempt. Radio and choice options need one of their words in the evidence.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from functools import cached_property

from .models import FieldType, FormField

__all__ = ["value_supported"]

# Answers that judge the evidence instead of copying it ("Married: yes" -> checked).
_JUDGMENT_WORDS = frozenset({"yes", "no", "y", "n", "true", "false", "on", "off", "x", "checked", "unchecked"})
# Radio export values without meaning of their own ("Choice1", "0", "Opt 2"); the
# model picks those by position in the attached form.
_GENERIC_EXPORT = re.compile(r"(?:choice|option|opt|item|button|btn|radio|check|box|value|val|group|kid)?[ _-]?[0-9]{0,3}")
_STOPWORDS = frozenset(
    {"a", "an", "the", "of", "and", "or", "to", "in", "on", "for", "with", "at", "by", "from", "as", "is", "are", "if", "not"}
)

_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc", "puerto rico": "pr",
}  # fmt: skip
_ADDRESS_WORDS = {
    "street": "st", "avenue": "ave", "av": "ave", "road": "rd", "boulevard": "blvd", "drive": "dr",
    "lane": "ln", "court": "ct", "place": "pl", "terrace": "ter", "highway": "hwy", "parkway": "pkwy",
    "circle": "cir", "square": "sq", "trail": "trl", "suite": "ste", "apartment": "apt",
    "building": "bldg", "floor": "fl", "room": "rm", "department": "dept", "number": "no",
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw",
    "southeast": "se", "southwest": "sw", "mount": "mt", "saint": "st", "fort": "ft",
    "junior": "jr", "senior": "sr", "usa": "us",
}  # fmt: skip
# Both sides of a comparison are mapped the same way, so a collision such as
# "court"/"Connecticut" -> "ct" only makes the check a little more lenient.
_PHRASES = {
    ("united", "states", "of", "america"): "us",
    ("united", "states"): "us",
    ("post", "office"): "po",
    **{tuple(name.split()): code for name, code in _STATES.items() if " " in name},
}
_WORDS = {**{name: code for name, code in _STATES.items() if " " not in name}, **_ADDRESS_WORDS}
_MAX_PHRASE = max(len(p) for p in _PHRASES)

_LETTERS = re.compile(r"[^\W\d_]+")
# A number written with separators: 123-45-6789, (555) 123-4567, 1,234.56.
_DIGIT_GROUP = re.compile(r"[0-9]+(?:[\s\-./,()+']{1,3}[0-9]+)*")
_ACRONYM = re.compile(r"\b(?:[^\W\d_]\.){2,}")  # "u.s.a." -> "usa", "p.o." -> "po"
_URLISH = re.compile(r"\S*(?:@|://|www\.)\S*")  # emails and web addresses: order matters

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
    "oct": 10, "nov": 11, "dec": 12,
}  # fmt: skip
_MONTH = r"(?P<mon>" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"
_WEEKDAY = r"(?:(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*\.?,?\s+)?"
_DAY = r"(?P<d>[0-9]{1,2})(?:st|nd|rd|th)?"
_YEAR = r"(?:1[89]|2[01])[0-9]{2}"  # 1800-2199: other 4-digit numbers are not years here
_FULL_DATES = tuple(
    re.compile(p)
    for p in (
        rf"(?<![0-9])(?P<y>{_YEAR})(?P<sep>[-/.])(?P<m>[0-9]{{1,2}})(?P=sep)(?P<d>[0-9]{{1,2}})(?![0-9])",
        rf"(?<![0-9])(?P<a>[0-9]{{1,2}})(?P<sep>[-/.])(?P<b>[0-9]{{1,2}})(?P=sep)(?P<y>{_YEAR}|[0-9]{{2}})(?![0-9])",
        rf"\b{_WEEKDAY}{_MONTH}\s+{_DAY},?\s+(?P<y>{_YEAR})\b",
        rf"\b{_WEEKDAY}{_DAY}\s+(?:of\s+)?{_MONTH},?\s+(?P<y>{_YEAR})\b",
        r"(?<![0-9])(?P<c>[0-9]{8})(?![0-9])",
    )
)
_MONTH_YEARS = tuple(
    re.compile(p)
    for p in (
        rf"\b{_MONTH},?\s+(?P<y>{_YEAR})\b",
        rf"(?<![0-9])(?P<m>[0-9]{{1,2}})/(?P<y>{_YEAR})(?![0-9])",
    )
)

DateKey = tuple[int, int, int]  # (year, month, day); day 0 for a month and year


def value_supported(f: FormField, raw: str, pdf_value: str, *, values: str, context: str) -> bool:
    """True if the evidence states the proposed value.

    `raw` is the model's value and `pdf_value` the checked value (they differ
    after reformatting, e.g. an ISO date rendered in the field's format).
    `values` is the text the evidence stands for; `context` also includes
    structured key paths, which can name an option ("employment.self_employed").
    """
    if f.type is FieldType.CHECKBOX:
        return True
    if f.type in (FieldType.RADIO, FieldType.CHOICE):
        options = [o for o in f.options if o.value == pdf_value]
        if options:
            texts = [t for o in options for t in (o.value, o.label) if t and t.strip()]
            return _option_supported(texts, _Evidence(context), radio=f.type is FieldType.RADIO)
    evidence = _Evidence(values)
    return any(_text_supported(v, evidence) for v in dict.fromkeys((raw, pdf_value)))


def _option_supported(texts: list[str], evidence: _Evidence, *, radio: bool) -> bool:
    meaningful = [t for t in texts if not _is_judgment(t) and not (radio and _GENERIC_EXPORT.fullmatch(_canon(t).strip()))]
    if not meaningful:
        return True
    for text in meaningful:
        canon = _canon(text)
        words = _words(canon)
        significant = [w for w in words if w.text not in _STOPWORDS] or words
        if any(evidence.has_word(w) for w in significant) or any(evidence.has_digits(d) for d in _digit_groups(canon)):
            return True
    return False


def _text_supported(value: str, evidence: _Evidence) -> bool:
    """Every word and number of the value is in the evidence; dates may be
    written in another format ("01/02/1980" for "January 2, 1980")."""
    if _is_judgment(value):
        return True
    canon = _canon(value)
    if _words_and_numbers_supported(canon, evidence):
        return True
    found, rest = _find_dates(canon)
    return bool(found) and all(evidence.has_date(keys) for keys in found) and _words_and_numbers_supported(rest, evidence)


def _words_and_numbers_supported(canon: str, evidence: _Evidence) -> bool:
    for match in _URLISH.finditer(canon):
        if match.group().strip(".,;:!?()[]<>\"'") not in evidence.text:
            return False
    return all(evidence.has_digits(d) for d in _digit_groups(canon)) and all(evidence.has_word(w) for w in _words(canon))


def _is_judgment(text: str) -> bool:
    return " ".join(_LETTERS.findall(_canon(text))) in _JUDGMENT_WORDS


# --------------------------------------------------------------------------- #
# Normalization and tokens
# --------------------------------------------------------------------------- #


def _canon(text: str) -> str:
    """Comparison form: casefolded, accents removed from Latin letters, ASCII
    digits, "-" for every dash, dotted acronyms joined, "&" spelled "and"."""
    out: list[str] = []
    for ch in unicodedata.normalize("NFKD", text):
        if unicodedata.combining(ch) and out and out[-1].isascii():
            continue  # "é" -> "e", but keep marks that distinguish letters in other scripts
        if ch.isdecimal() and not ch.isascii():
            ch = str(unicodedata.decimal(ch))
        elif ch == "\u2212" or unicodedata.category(ch) == "Pd":
            ch = "-"
        out.append(ch)
    text = _ACRONYM.sub(lambda m: m.group().replace(".", ""), "".join(out).casefold())
    return text.replace("&", " and ")


@dataclass(frozen=True)
class _Word:
    text: str
    initial: bool  # written as a single letter, so it may stand for a longer word


def _words(canon: str) -> list[_Word]:
    """Letter runs with multi-word names and address words abbreviated."""
    tokens = _LETTERS.findall(canon)
    words: list[_Word] = []
    i = 0
    while i < len(tokens):
        for size in range(min(_MAX_PHRASE, len(tokens) - i), 1, -1):
            code = _PHRASES.get(tuple(tokens[i : i + size]))
            if code is not None:
                words.append(_Word(code, initial=False))
                i += size
                break
        else:
            words.append(_Word(_WORDS.get(tokens[i], tokens[i]), initial=len(tokens[i]) == 1))
            i += 1
    return words


def _digit_groups(canon: str) -> list[str]:
    return [re.sub(r"[^0-9]", "", m.group()) for m in _DIGIT_GROUP.finditer(canon)]


class _Evidence:
    def __init__(self, text: str):
        self.text = _canon(text)

    @cached_property
    def _word_set(self) -> set[str]:
        return {w.text for w in _words(self.text)}

    @cached_property
    def _runs(self) -> list[list[str]]:
        return [re.findall(r"[0-9]+", m.group()) for m in _DIGIT_GROUP.finditer(self.text)]

    @cached_property
    def _dates(self) -> set[DateKey]:
        return set().union(*_find_dates(self.text)[0])

    def has_word(self, word: _Word) -> bool:
        if word.text in self._word_set:
            return True
        if word.initial and any(w.startswith(word.text) for w in self._word_set):
            return True
        return not word.text.isascii() and word.text in self.text  # scripts written without spaces

    def has_digits(self, digits: str) -> bool:
        """True if consecutive numbers of one group spell `digits` exactly
        ("123456789" in "SSN 123-45-6789", or standard SSN/EIN/ZIP+4/phone
        parts of an unhyphenated 9- or 10-digit number; "1" is not in "2019")."""
        if not digits:
            return True
        for runs in self._runs:
            for start in range(len(runs)):
                joined = ""
                for run in runs[start:]:
                    joined += run
                    if len(joined) >= len(digits):
                        break
                if joined == digits:
                    return True
            for run in runs:
                if len(run) == 9 and digits in (run[:3], run[3:5], run[5:], run[:2], run[2:], run[:5], run[5:]):
                    return True
                if len(run) == 10 and digits in (run[:3], run[3:6], run[6:], run[1:4], run[4:]):
                    return True
        return False

    def has_date(self, keys: set[DateKey]) -> bool:
        for year, month, day in keys:
            if day and (year, month, day) in self._dates:
                return True
            if not day and any((y, m) == (year, month) for y, m, _ in self._dates):
                return True
        return False


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #


def _find_dates(canon: str) -> tuple[list[set[DateKey]], str]:
    """The dates in `canon` (each as its possible readings), plus the text without them.

    Numeric dates keep both month/day orders and both centuries of a 2-digit
    year; a match that isn't a real date in any reading is left as text.
    """
    found: list[set[DateKey]] = []
    chars = list(canon)
    for patterns, full in ((_FULL_DATES, True), (_MONTH_YEARS, False)):
        for pattern in patterns:
            for match in pattern.finditer("".join(chars)):
                keys = _readings(match.groupdict(), full)
                if keys:
                    found.append(keys)
                    chars[match.start() : match.end()] = " " * (match.end() - match.start())
    return found, "".join(chars)


def _readings(g: dict[str, str | None], full: bool) -> set[DateKey]:
    candidates: list[tuple[int, int, int]] = []
    if not full:
        month = _MONTHS[g["mon"]] if g.get("mon") else int(g["m"] or 0)
        return {(int(g["y"] or 0), month, 0)} if 1 <= month <= 12 else set()
    if g.get("c"):
        c = g["c"]
        candidates = [(int(c[4:]), int(c[:2]), int(c[2:4])), (int(c[4:]), int(c[2:4]), int(c[:2])), (int(c[:4]), int(c[4:6]), int(c[6:]))]
    elif g.get("a"):
        a, b, y = int(g["a"]), int(g["b"]), g["y"]
        years = [int(y)] if len(y) == 4 else [1900 + int(y), 2000 + int(y)]
        candidates = [(year, m, d) for year in years for m, d in ((a, b), (b, a))]
    elif g.get("mon"):
        candidates = [(int(g["y"]), _MONTHS[g["mon"]], int(g["d"]))]
    else:
        candidates = [(int(g["y"]), int(g["m"]), int(g["d"]))]
    return {key for key in candidates if _real(key)}


def _real(key: DateKey) -> bool:
    if not 1800 <= key[0] <= 2199:
        return False
    try:
        date(*key)
    except ValueError:
        return False
    return True
