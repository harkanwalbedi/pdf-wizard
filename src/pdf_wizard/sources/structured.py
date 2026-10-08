"""Structured data (--data): one JSON or CSV record, flattened to `key.path = value` lines."""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Iterator
from typing import Any

from ..errors import InvalidInputError
from ..models import Source, SourceKind
from .common import (
    Context,
    clean_text,
    decode_utf8,
    escape_newlines,
    has_binary_bytes,
    has_disallowed_controls,
    invalid,
    is_image,
    is_pdf,
    plural,
)

_SNIFF_CHARS = 4096
_DELIMITERS = ",;\t|"
# JSON keys with these characters (or empty / padded keys) are written as ["..."]
# so that paths stay unambiguous: {"a.b": 1} and {"a": {"b": 2}} differ.
_UNSAFE_KEY_CHARS = re.compile(r'[.\[\]"\\=\x00-\x1f\x7f-\x9f\u2028\u2029]')


def load_structured(data: bytes, ctx: Context, *, row: int | None) -> Source:
    """Loads one record from a JSON or CSV file. `row` is 1-based."""
    if row is not None and row < 1:
        raise InvalidInputError("Row numbers start at 1.")
    if is_pdf(data) or is_image(data) or has_binary_bytes(data):
        raise invalid(ctx, "is not a JSON or CSV file")
    text = decode_utf8(data)
    if text is None:
        raise invalid(ctx, "is not valid UTF-8 text")
    if has_disallowed_controls(text):
        raise invalid(ctx, "is not a JSON or CSV file")

    if text.lstrip()[:1] in ("{", "["):
        keypaths = _json_record(text, ctx, row)
    else:
        keypaths = _csv_record(text, ctx, row)
    if not keypaths:
        raise invalid(ctx, "has no values in the selected record")

    # One line per key path: line breaks in keys and values are escaped.
    body = "\n".join(f"{escape_newlines(path)} = {escape_newlines(value)}" for path, value in keypaths.items())
    if len(body) > ctx.limits.max_source_chars:  # structured data is never truncated
        raise invalid(
            ctx, f"has too much data (limit {ctx.limits.max_source_chars:,} characters); remove unused fields"
        )
    return Source(id=ctx.sid, kind=SourceKind.STRUCTURED, name=ctx.name, text=body, keypaths=keypaths)


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not allowed")


def _json_record(text: str, ctx: Context, row: int | None) -> dict[str, str]:
    try:
        # Numbers stay strings, so "1.50" and "00501"-style values keep their
        # exact form and huge integers can't hit int-conversion limits.
        doc = json.loads(text, parse_int=str, parse_float=str, parse_constant=_reject_constant)
    except RecursionError as exc:
        raise invalid(ctx, "is nested too deeply") from exc
    except ValueError as exc:
        where = f" (line {exc.lineno}, column {exc.colno})" if isinstance(exc, json.JSONDecodeError) else ""
        raise invalid(ctx, f"is not valid JSON{where}") from exc

    if isinstance(doc, dict):
        _record_index(ctx, row, 1)
        record, depth = doc, 1
    else:  # a list: the first character was "{" or "["
        record, depth = _pick_json_record(doc, ctx, row), 2
    try:
        return _flatten(record, ctx, depth)
    except RecursionError as exc:  # only with a very large configured max_json_depth
        raise invalid(ctx, "is nested too deeply") from exc


def _pick_json_record(items: list[Any], ctx: Context, row: int | None) -> dict[str, Any]:
    if not items:
        raise invalid(ctx, "has no records")
    if not all(isinstance(item, dict) for item in items):
        raise invalid(ctx, "must contain a JSON object or a list of objects")
    return items[_record_index(ctx, row, len(items))]


def _flatten(record: dict[str, Any], ctx: Context, base_depth: int) -> dict[str, str]:
    """{"a": {"b": [{"c": 1}]}} -> {"a.b[0].c": "1"}. Nulls and empty strings are skipped."""
    limits = ctx.limits
    out: dict[str, str] = {}
    keys = 0

    def walk(node: dict[str, Any] | list[Any], path: str, depth: int) -> None:
        nonlocal keys
        if depth > limits.max_json_depth:
            raise invalid(ctx, f"is nested too deeply (limit {limits.max_json_depth} levels)")
        children: Iterator[tuple[str | int, Any]] = iter(node.items() if isinstance(node, dict) else enumerate(node))
        for key, value in children:
            keys += 1
            if keys > limits.max_json_keys:
                raise invalid(ctx, f"has too many keys (limit {limits.max_json_keys:,})")
            child = _child_path(path, key)
            if isinstance(value, (dict, list)):
                walk(value, child, depth + 1)
            elif (scalar := _scalar(value)) is not None:
                out[child] = scalar

    walk(record, "", base_depth)
    return out


def _child_path(path: str, key: str | int) -> str:
    if isinstance(key, int):
        return f"{path}[{key}]"
    if key and key == key.strip() and not _UNSAFE_KEY_CHARS.search(key):
        return f"{path}.{key}" if path else key
    return f"{path}[{json.dumps(key)}]"


def _scalar(value: Any) -> str | None:
    if value is None:
        return None
    if value is True:
        return "true"
    if value is False:
        return "false"
    text = clean_text(str(value)).strip()  # strings and number lexemes
    return text or None


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #


def _csv_record(text: str, ctx: Context, row: int | None) -> dict[str, str]:
    try:
        delimiter = csv.Sniffer().sniff(text[:_SNIFF_CHARS], delimiters=_DELIMITERS).delimiter
    except Exception:  # the sniffer is a heuristic; fall back to commas
        delimiter = ","
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    try:
        header = next((r for r in reader if r), None)  # skip leading blank lines
        if header is None:
            raise invalid(ctx, "has no header row")
        names = _header_names(header, ctx)
        first, wanted, count = _scan_rows(reader, row)
    except csv.Error as exc:  # e.g. a field over csv.field_size_limit()
        raise invalid(ctx, "could not be read as CSV") from exc

    if count == 0:
        raise invalid(ctx, "has no data rows")
    index = _record_index(ctx, row, count)
    chosen = first if index == 0 else wanted
    if chosen is None:  # not reachable: rows up to `count` were all seen
        raise invalid(ctx, "has no data rows")
    if any(v.strip() for v in chosen[len(names):]):
        raise InvalidInputError(
            f"Row {index + 1} of '{ctx.name}' has more values than there are columns in the header."
        )
    record: dict[str, str] = {}
    for name, value in zip(names, chosen):
        if value := clean_text(value).strip():
            record[name] = value
    return record


def _header_names(header: list[str], ctx: Context) -> list[str]:
    if len(header) > ctx.limits.max_json_keys:
        raise invalid(ctx, f"has too many columns (limit {ctx.limits.max_json_keys:,})")
    names = [" ".join(clean_text(h).split()) for h in header]
    if not all(names):
        raise invalid(ctx, "has an empty column name in its header row")
    if len(set(names)) != len(names):
        raise invalid(ctx, "has duplicate column names in its header row")
    return names


def _scan_rows(
    reader: Iterator[list[str]], row: int | None
) -> tuple[list[str] | None, list[str] | None, int]:
    """Returns (row 1, row number `row`, number of records) without keeping every row in memory.

    Blank lines are skipped, and rows with only empty values at the end of the
    file (spreadsheet leftovers) are not counted as records.
    """
    first: list[str] | None = None
    wanted: list[str] | None = None
    index = count = 0
    for values in reader:
        if not values:
            continue
        index += 1
        if index == 1:
            first = values
        if index == row:
            wanted = values
        if any(v.strip() for v in values):
            count = index
    return first, wanted, count


# --------------------------------------------------------------------------- #
# Record selection
# --------------------------------------------------------------------------- #


def _record_index(ctx: Context, row: int | None, count: int) -> int:
    """0-based index of the record to use; `row` is 1-based.

    The CLI and the web app pass one row number for every data file, so it
    only applies to files with several records.
    """
    if count == 1:
        if row is not None and row > 1:
            ctx.warn("only one record; the row number was not used.")
        return 0
    if row is None:
        raise InvalidInputError(f"'{ctx.name}' has {count:,} records; choose one with --row (1-{count}).")
    if row > count:
        raise InvalidInputError(f"Row {row} is out of range: '{ctx.name}' has {plural(count, 'record')}.")
    return row - 1
