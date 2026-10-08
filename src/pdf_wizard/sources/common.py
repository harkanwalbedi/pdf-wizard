"""Types and helpers shared by the source loaders.

File names and contents are untrusted and may contain personal data: they never
go into warnings or logs. Warnings refer to sources by ID ("S2").
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal

from ..config import MB, Limits
from ..errors import InvalidInputError
from ..models import Source

Role = Literal["data", "doc", "reference", "note"]
ROLES = frozenset({"data", "doc", "reference", "note"})

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"
PDF_MAGIC = b"%PDF-"

NAME_MAX_CHARS = 80
_NAME_FALLBACK = "file"
_NAME_DISALLOWED = re.compile(r"[^A-Za-z0-9 ._()\-]")
_PATH_SEPARATORS = re.compile(r"[/\\]")

# Control characters (Unicode category Cc: C0, DEL, C1).
_CONTROLS_EXCEPT_NL_TAB = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_CONTROLS_EXCEPT_TEXT_WS = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f-\x9f]")  # keeps \t \n \f \r
_BINARY_BYTES = re.compile(rb"[\x00-\x08\x0b\x0e-\x1f\x7f]")  # C0 controls (except \t \n \f \r) and DEL

# Every line boundary of str.splitlines(), as a visible escape. CR and CRLF are
# normalized to LF before the table is applied.
_LINE_BREAK_ESCAPES = str.maketrans(
    {
        "\n": "\\n",
        "\r": "\\r",
        "\v": "\\v",
        "\f": "\\f",
        "\x1c": "\\x1c",
        "\x1d": "\\x1d",
        "\x1e": "\\x1e",
        "\x85": "\\x85",
        "\u2028": "\\u2028",
        "\u2029": "\\u2029",
    }
)


@dataclass(frozen=True)
class SourceInput:
    """One user input. `name` and `content` are untrusted and kept out of repr()."""

    role: Role
    name: str = field(repr=False)  # original file name (untrusted); "" for notes
    content: bytes = field(repr=False)  # file bytes; for notes, UTF-8 text
    row: int | None = None  # data only: 1-based record for CSV files / JSON arrays


@dataclass
class LoadedSources:
    sources: list[Source]  # S0 first, then S1.. in input order
    warnings: list[str] = field(default_factory=list)  # safe: no values or file names


@dataclass
class Context:
    """What a loader needs to know about the source it is building."""

    sid: str  # "S3"
    name: str  # sanitized display name
    limits: Limits
    warnings: list[str]  # shared by all sources; messages must not contain values or names

    def warn(self, message: str) -> None:
        self._add(f"{self.sid}: {message}")

    def truncate(self, text: str, *, message: str | None = None) -> str:
        """Cuts `text` to limits.max_source_chars, warning (once) if it was longer."""
        limit = self.limits.max_source_chars
        if len(text) <= limit:
            return text
        self.mark_truncated(message)
        return text[:limit]

    def mark_truncated(self, message: str | None = None) -> None:
        if message is None:
            self._add(f"{self.sid} was truncated to {self.limits.max_source_chars:,} characters.")
        else:
            self.warn(message)

    def _add(self, warning: str) -> None:
        if warning not in self.warnings:
            self.warnings.append(warning)


def display_name(raw: str) -> str:
    """A display name for an untrusted file name; it is shown in the model prompt.

    Basename only, control characters dropped, accents removed, and anything
    outside [A-Za-z0-9 ._()-] replaced by "_". At most 80 characters.
    """
    name = _PATH_SEPARATORS.split((raw or "")[-1024:])[-1]
    name = "".join(ch for ch in name if unicodedata.category(ch) not in ("Cc", "Cf"))
    name = unicodedata.normalize("NFKD", name)
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    name = _NAME_DISALLOWED.sub("_", name)
    name = re.sub(r"_{2,}", "_", name)
    name = re.sub(r" {2,}", " ", name)
    name = re.sub(r"\.{2,}", ".", name)
    name = name.strip(" .")
    if len(name) > NAME_MAX_CHARS:
        stem, dot, ext = name.rpartition(".")
        if dot and stem and len(ext) <= 10 and ext.isalnum():  # keep a short extension
            name = stem[: NAME_MAX_CHARS - len(ext) - 1].rstrip(" .") + "." + ext
        else:
            name = name[:NAME_MAX_CHARS].rstrip(" .")
    return name or _NAME_FALLBACK


def decode_utf8(data: bytes) -> str | None:
    """Strict UTF-8 (a leading BOM is dropped); None if the bytes aren't UTF-8."""
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def clean_text(text: str) -> str:
    """Normalizes newlines to \\n and drops control characters other than \\n and \\t."""
    return _CONTROLS_EXCEPT_NL_TAB.sub("", normalize_newlines(text))


def has_disallowed_controls(text: str) -> bool:
    """True if `text` has control characters other than \\t \\n \\f \\r."""
    return _CONTROLS_EXCEPT_TEXT_WS.search(text) is not None


def has_binary_bytes(data: bytes) -> bool:
    """True if `data` has NUL or other C0 control bytes that plain text doesn't contain."""
    return _BINARY_BYTES.search(data) is not None


def escape_newlines(value: str) -> str:
    """Renders `value` on one line, for "key = value" lines that are parsed line by line.

    Every character that str.splitlines() treats as a line boundary is escaped,
    so a value can't start a line of its own: CR, LF and CRLF become "\\n", the
    others visible escapes such as "\\u2028".
    """
    return normalize_newlines(value).translate(_LINE_BREAK_ESCAPES)


def is_pdf(data: bytes) -> bool:
    return PDF_MAGIC in data[:1024]


def is_image(data: bytes) -> bool:
    return data.startswith(PNG_MAGIC) or data.startswith(JPEG_MAGIC)


def format_bytes(n: int) -> str:
    if n >= MB:
        return f"{n / MB:g} MB"
    if n >= 1024:
        return f"{n / 1024:g} KB"
    return f"{n:,} bytes"


def plural(n: int, word: str) -> str:
    return f"{n:,} {word}" + ("" if n == 1 else "s")


def invalid(ctx: Context, problem: str) -> InvalidInputError:
    """An InvalidInputError about one file, e.g. invalid(ctx, "is not valid JSON")."""
    return InvalidInputError(f"'{ctx.name}' {problem}.")
