"""Detection and masking of sensitive personal data.

These checks are a backstop: a value is treated as sensitive if the model says
so OR the patterns here match, so a missed flag still gets masked. The same
patterns scrub free text (log messages, warnings, evidence) before it is shown
or stored.
"""

from __future__ import annotations

import calendar
import re
import unicodedata
from collections.abc import Iterable

from .models import FieldType, FormField

# Dash-like separators, dots, slashes, and whitespace are allowed inside digit
# runs ("1234 5678", "123-45-6789", "123.45.6789", "123/45/6789").
_SEP = r"[\s\-./\u2010-\u2015\u2212]"

# 8+ digits, optionally with single separators: account, card, SSN/ITIN/EIN
# shapes. The two lookbehinds make every match start at the beginning of a
# run, which keeps the scan linear on adversarial input.
_DIGIT_RUN = re.compile(rf"(?<!\d)(?<!\d{_SEP})\d(?:{_SEP}?\d){{7,}}(?!\d)")
_SSN = re.compile(rf"(?<!\d)(?<!\d{_SEP})\d{{3}}{_SEP}?\d{{2}}{_SEP}?\d{{4}}(?!\d)(?!{_SEP}\d)")
_EIN = re.compile(r"(?<!\d)\d{2}-\d{7}(?!\d)")
_CARD = re.compile(rf"(?<!\d)(?<!\d{_SEP})\d(?:{_SEP}?\d){{12,18}}(?!\d)(?!{_SEP}\d)")
# Plain decimal numbers ("1234567.89") are amounts, not identifiers.
_PLAIN_DECIMAL = re.compile(r"\d+\.\d+")

# Possessive quantifiers and the start-of-run lookbehind avoid catastrophic
# backtracking; the top-level domain is checked in `_email_spans`.
_EMAIL = re.compile(r"(?<![\w.%+-])[\w.%+-]++@[\w-]++(?:\.[\w-]++)++")
_TLD = re.compile(r"[A-Za-z]{2,63}")

# Numeric dates with matching separators ("01/02/1980", "1980-01-02"). Valid
# calendar dates in 1900-2099 are shielded before scanning for digit runs so
# ordinary dates ("Date signed: 2024-01-02") are not mistaken for 8-digit IDs.
_NUMERIC_DATE = re.compile(
    r"(?<![\d./\-])(?:"
    r"(?P<a>\d{1,2})(?P<s1>[./\-])(?P<b>\d{1,2})(?P=s1)(?P<y1>\d{4}|\d{2})"
    r"|(?P<y2>\d{4})(?P<s2>[./\-])(?P<m>\d{1,2})(?P=s2)(?P<d>\d{1,2})"
    r")(?!\d|[./\-]\d)"
)

_MONTH_NAMES = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)  # fmt: skip
_MONTH_LOOKUP: dict[str, int] = {}
for _idx, _name in enumerate(_MONTH_NAMES, start=1):
    _MONTH_LOOKUP[_name] = _idx
    _MONTH_LOOKUP[_name[:3]] = _idx
_MONTH_LOOKUP["sept"] = 9

_MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?"
    r"|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
)
_DAY = r"[0-3]?\d(?:st|nd|rd|th)?"
_YEAR = r"(?:\d{4}|\d{2})"
_NAMED_DATE = (
    rf"(?:{_MONTH}[\s\-]{{1,3}}{_DAY}(?:,?[\s\-]{{1,3}}{_YEAR})?"
    rf"|{_DAY}(?:[\s\-]{{1,3}}of)?[\s\-]{{1,3}}{_MONTH}(?:,?[\s\-]{{1,3}}{_YEAR})?)"
)
_ANY_DATE_SRC = rf"(?:\d{{1,2}}[./\-]\d{{1,2}}[./\-]{_YEAR}|\d{{4}}[./\-]\d{{1,2}}[./\-]\d{{1,2}}|{_NAMED_DATE})"
_ANY_DATE = re.compile(_ANY_DATE_SRC, re.IGNORECASE | re.ASCII)

_NAMED_DATE_PARTS = re.compile(
    rf"(?:(?P<m1>{_MONTH})[\s\-]{{1,3}}(?P<d1>{_DAY})(?:,?[\s\-]{{1,3}}(?P<y1>{_YEAR}))?"
    rf"|(?P<d2>{_DAY})(?:[\s\-]{{1,3}}of)?[\s\-]{{1,3}}(?P<m2>{_MONTH})(?:,?[\s\-]{{1,3}}(?P<y2>{_YEAR}))?)",
    re.IGNORECASE | re.ASCII,
)

# Dates anchored on a nearby birth-date keyword ("DOB 01/02/1980",
# "Date of birth: January 2, 1980", "Born 12/31/1990").
_BIRTH_KEYWORD = (
    r"(?:d\.?\s?o\.?\s?b\.?"
    r"|date[\s\-_]*(?:(?:and|&)[\s\-_]*place[\s\-_]*)?of[\s\-_]*birth"
    r"|birth[\s\-_]*date|birthday|born)"
)
_BIRTH_DATE = re.compile(
    rf"(?<![^\W\d_]){_BIRTH_KEYWORD}(?![^\W\d_])\D{{0,40}}?(?<!\w)(?P<date>{_ANY_DATE_SRC})(?!\w)",
    re.IGNORECASE,
)

# Field labels/names that mark a value as sensitive, matched on whole words
# after `_label_words` splits camelCase and punctuation ("applicantDOB",
# "f1_ssn[0]", "Driver's License #", "D.O.B.").
_LABEL_TERMS = re.compile(
    r"(?<![a-z0-9])(?:"
    r"ssn|social security|soc sec|itin|ein|tin|tax ?id(?:entification)?|taxpayer ?id(?:entification)?"
    r"|passport|drivers? ?licen[cs]e|driving ?licen[cs]e"
    r"|account|acct|routing|iban|card ?(?:no|num|nbr|number)|(?:credit|debit) ?card|cvv\d?|cvc"
    r"|pin|password|passcode|date ?of ?birth|birth ?date|birthday|d ?o ?b|born"
    r")(?: ?(?:no|num|nbr|number|code))?(?![a-z0-9])"
)

_GROUP = re.compile(r"\d+|[^\W\d_]+")
_GAP = re.compile(r"[\s\-./,\u2010-\u2015\u2212]{1,3}\Z")


def is_sensitive(field: FormField, value: str) -> bool:
    """True if `value` looks like sensitive data or `field` is labeled as holding it.

    Pattern checks (all field types): SSN/ITIN, EIN, card numbers (Luhn), email
    addresses, birth-date phrases, and runs of 8+ digits (account numbers) that
    are not plain dates or decimal amounts. Label checks (text and choice
    fields): SSN, tax ID, passport, account, card, PIN, date of birth...
    Checkbox and radio states ("checked", "married") are not identifiers, so
    their labels are not checked.
    """
    if value and _has_sensitive_pattern(value):
        return True
    if field.password or (field.format_hint or "").strip().lower() == "ssn":
        return True
    if field.type in (FieldType.CHECKBOX, FieldType.RADIO):
        return False
    return any(_LABEL_TERMS.search(_label_words(text)) for text in (field.label, field.name) if text)


def mask_value(value: str) -> str:
    """Masks letters and digits, keeping separators: "123-45-6789" -> "***-**-6789".

    The last 4 letters/digits stay visible only if the value has at least 8 and
    is not date-shaped (for a date, the last 4 characters are the year or the
    month and day). Short values and dates are fully masked. Invisible format
    characters (Cf) are dropped when other characters are present. Non-empty
    input is never returned unchanged.
    """
    if not value:
        return value
    visible, indices = _visible_view(value)
    work = visible if (indices is not None and visible) else value
    maskable = [i for i, ch in enumerate(work) if _is_maskable(ch)]
    if not maskable:
        masked = "*" * len(work)
        return masked if masked != value else masked + "*"
    is_date = bool(_ANY_DATE.fullmatch(work.strip()))
    hidden = set(maskable if (is_date or len(maskable) < 8) else maskable[:-4])
    return "".join("*" if i in hidden else ch for i, ch in enumerate(work))


def scrub_text(text: str) -> str:
    """Masks SSN-, card-, account-number-, birth-date-, and email-like spans in free text."""
    if not text:
        return text
    visible, indices = _visible_view(text)
    spans = _email_spans(visible)
    spans.extend(m.span("date") for m in _BIRTH_DATE.finditer(visible))
    shielded = _shield_dates(visible)
    for pattern in (_SSN, _EIN, _DIGIT_RUN):
        for m in pattern.finditer(shielded):
            if not _PLAIN_DECIMAL.fullmatch(m.group()):
                spans.append(m.span())
    for m in _CARD.finditer(shielded):
        if not _PLAIN_DECIMAL.fullmatch(m.group()) and luhn_valid(m.group()):
            spans.append(m.span())
    if not spans:
        return text
    if indices is not None:
        spans = [(indices[s], indices[e - 1] + 1) for s, e in spans if e > s]
    return _mask_spans(text, spans)


def luhn_valid(number: str) -> bool:
    """True if the digits in `number` pass the Luhn checksum used by payment cards."""
    digits = [int(ch) for ch in number if ch.isdecimal()]
    if not digits:
        return False
    total = 0
    for position, digit in enumerate(reversed(digits)):
        if position % 2:
            digit = digit * 2 - 9 if digit > 4 else digit * 2
        total += digit
    return total % 10 == 0


class ValueRedactor:
    """Masks occurrences of known sensitive field values across free text.

    Matches values regardless of separator style or letter case (so a sensitive
    SSN "123-45-6789", passport "X1234567", or DOB "01/02/1980" is redacted
    even when quoted as "123.45.6789", "x 1234567", or "January 2, 1980" in
    another field's evidence or notes). Values with fewer than 3 letters/digits
    are ignored to avoid masking common short words or numbers.
    """

    def __init__(self, values: Iterable[str]) -> None:
        self._keys: set[str] = set()
        self._prefixes: set[str] = set()
        self._literals: list[str] = []
        seen_literals: set[str] = set()

        for raw in values:
            if not raw:
                continue
            clean, _ = _visible_view(raw.strip())
            if not clean:
                continue
            groups = [m.group().casefold() for m in _GROUP.finditer(clean)]
            key = "".join(groups)
            if len(key) < 3:
                continue
            if len(key) <= 64:
                self._add_key(key)
            for dkey in _date_keys(clean):
                if 6 <= len(dkey) <= 64:
                    self._add_key(dkey)
            lit = clean.casefold()
            if 3 <= len(lit) <= 128 and lit not in seen_literals:
                seen_literals.add(lit)
                self._literals.append(lit)

    def _add_key(self, key: str) -> None:
        self._keys.add(key)
        for i in range(1, len(key) + 1):
            self._prefixes.add(key[:i])

    def __call__(self, text: str) -> str:
        if not text or (not self._keys and not self._literals):
            return text
        visible, indices = _visible_view(text)
        spans: list[tuple[int, int]] = []

        if self._keys:
            tokens = [(m.start(), m.end(), m.group().casefold()) for m in _GROUP.finditer(visible)]
            n = len(tokens)
            for i in range(n):
                acc = ""
                for j in range(i, n):
                    if j > i:
                        gap = visible[tokens[j - 1][1] : tokens[j][0]]
                        if gap != "" and not _GAP.match(gap):
                            break
                    acc += tokens[j][2]
                    if acc not in self._prefixes:
                        break
                    if acc in self._keys:
                        spans.append((tokens[i][0], tokens[j][1]))

        if self._literals:
            lowered = visible.casefold()
            vlen = len(visible)
            if len(lowered) == vlen:
                for lit in self._literals:
                    llen = len(lit)
                    pos = 0
                    while (found := lowered.find(lit, pos)) != -1:
                        end = found + llen
                        if not _glued(visible, found, end):
                            spans.append((found, end))
                        pos = found + 1

        if not spans:
            return text
        if indices is not None:
            spans = [(indices[s], indices[e - 1] + 1) for s, e in spans if e > s]
        return _mask_spans(text, spans)


def _has_sensitive_pattern(text: str) -> bool:
    visible, _ = _visible_view(text)
    if _email_spans(visible) or _BIRTH_DATE.search(visible):
        return True
    shielded = _shield_dates(visible)
    if any(not _PLAIN_DECIMAL.fullmatch(m.group()) for m in _SSN.finditer(shielded)):
        return True
    if _EIN.search(shielded):
        return True
    if any(not _PLAIN_DECIMAL.fullmatch(m.group()) and luhn_valid(m.group()) for m in _CARD.finditer(shielded)):
        return True
    return any(not _PLAIN_DECIMAL.fullmatch(m.group()) for m in _DIGIT_RUN.finditer(shielded))


def _visible_view(text: str) -> tuple[str, list[int] | None]:
    """Drops Unicode format characters (Cf, e.g. zero-width space) with an index map."""
    if text.isascii():
        return text, None
    if not any(unicodedata.category(ch) == "Cf" for ch in text):
        return text, None
    chars: list[str] = []
    indices: list[int] = []
    for i, ch in enumerate(text):
        if unicodedata.category(ch) != "Cf":
            chars.append(ch)
            indices.append(i)
    return "".join(chars), indices


def _shield_dates(text: str) -> str:
    """Replaces valid numeric dates (1900-2099) with '#' of the same length."""

    def _replace(m: re.Match[str]) -> str:
        return "#" * len(m.group()) if _numeric_date_readings(m) else m.group()

    return _NUMERIC_DATE.sub(_replace, text)


def _numeric_date_readings(m: re.Match[str]) -> list[tuple[int, int, int]]:
    readings: list[tuple[int, int, int]] = []
    if m.group("y2") is not None:
        year = int(m.group("y2"))
        month = int(m.group("m"))
        day = int(m.group("d"))
        if _valid_ymd(year, month, day):
            readings.append((year, month, day))
        return readings

    a = int(m.group("a"))
    b = int(m.group("b"))
    y_str = m.group("y1")
    years = [int(y_str)] if len(y_str) == 4 else [1900 + int(y_str), 2000 + int(y_str)]
    for year in years:
        for month, day in ((a, b), (b, a)):
            if _valid_ymd(year, month, day) and (year, month, day) not in readings:
                readings.append((year, month, day))
    return readings


def _valid_ymd(year: int, month: int, day: int) -> bool:
    if not (1900 <= year <= 2099 and 1 <= month <= 12 and day >= 1):
        return False
    return day <= calendar.monthrange(year, month)[1]


def _date_keys(text: str) -> set[str]:
    readings: list[tuple[int, int, int]] = []
    if m := _NUMERIC_DATE.fullmatch(text):
        readings.extend(_numeric_date_readings(m))
    elif m := _NAMED_DATE_PARTS.fullmatch(text):
        m_raw = (m.group("m1") or m.group("m2") or "").rstrip(".").casefold()
        d_raw = m.group("d1") or m.group("d2") or ""
        y_raw = m.group("y1") or m.group("y2")
        month = _MONTH_LOOKUP.get(m_raw, 0)
        day = int(re.sub(r"\D", "", d_raw) or "0")
        if month and y_raw:
            years = [int(y_raw)] if len(y_raw) == 4 else [1900 + int(y_raw), 2000 + int(y_raw)]
            for year in years:
                if _valid_ymd(year, month, day):
                    readings.append((year, month, day))

    keys: set[str] = set()
    for year, month, day in readings:
        m1, m2 = str(month), f"{month:02d}"
        d1, d2 = str(day), f"{day:02d}"
        y4, y2 = f"{year:04d}", f"{year % 100:02d}"
        for mv in (m1, m2):
            for dv in (d1, d2):
                keys.update((f"{mv}{dv}{y4}", f"{mv}{dv}{y2}", f"{dv}{mv}{y4}", f"{dv}{mv}{y2}", f"{y4}{mv}{dv}"))
        m_full = _MONTH_NAMES[month - 1]
        m_names = {m_full, m_full[:3]} | ({"sept"} if month == 9 else set())
        suffix = "th" if 11 <= day % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        d_ord = f"{d1}{suffix}"
        for mn in m_names:
            for dv in (d1, d2, d_ord):
                for yv in (y4, y2):
                    keys.update((f"{mn}{dv}{yv}", f"{dv}{mn}{yv}", f"{dv}of{mn}{yv}"))
    return keys


def _glued(text: str, start: int, end: int) -> bool:
    if start > 0:
        left, first = _char_class(text[start - 1]), _char_class(text[start])
        if left and left == first:
            return True
    if end < len(text):
        last, right = _char_class(text[end - 1]), _char_class(text[end])
        if right and last == right:
            return True
    return False


def _char_class(ch: str) -> str:
    if ch.isdigit():
        return "d"
    if ch.isalpha():
        return "a"
    return ""


def _mask_spans(text: str, spans: list[tuple[int, int]]) -> str:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    for start, end in reversed(merged):
        text = text[:start] + mask_value(text[start:end]) + text[end:]
    return text


def _email_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in _EMAIL.finditer(text) if _TLD.fullmatch(m.group().rsplit(".", 1)[1])]


def _is_maskable(ch: str) -> bool:
    # Combining marks (accents, Indic vowel signs) carry information too.
    return ch.isalnum() or unicodedata.category(ch).startswith("M")


def _label_words(text: str) -> str:
    text = text.replace("'", "").replace("\u2019", "")
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)  # camelCase -> camel Case
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)  # SSNLast -> SSN Last
    text = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", text)
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()
