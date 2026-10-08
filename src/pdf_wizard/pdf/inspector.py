"""Form inspector: validates a PDF and builds the field catalog (stages 1-2).

The PDF is only read. Form JavaScript is never executed: format hints are
parsed from Acrobat's standard format scripts with regular expressions.
"""

from __future__ import annotations

import bisect
import hashlib
import io
import itertools
import logging
import math
import re
import xml.etree.ElementTree as ET
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from pypdf import PasswordType, PdfReader, PdfWriter
from pypdf.errors import LimitReachedError
from pypdf.generic import ArrayObject, DictionaryObject, IndirectObject, NameObject, NumberObject, StreamObject

from ..config import MB, Limits, legacy_heuristics_enabled
from ..errors import InvalidInputError, UnsupportedFormError
from ..models import FieldType, FormField, FormInfo, Option
from .limits import page_text, pypdf_limits

log = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF-"
MAX_TREE_DEPTH = 32
MAX_OPTIONS = 500
MAX_LABEL_CHARS = 200
MAX_SCRIPT_CHARS = 4096
MAX_XFA_BYTES = 2 * MB
# Label inference scores nearby page text for each search, so these bound its
# work per form to about MAX_LABEL_LOOKUPS x MAX_LABEL_LINES scores.
MAX_LABEL_LINES = 2_000  # text lines kept per page, from the top
MAX_LABEL_LOOKUPS = 2_000  # page-text label searches per form
MAX_LABELED_STATES = 20  # a widget with more on-states gets no page-text label

TOO_COMPLEX = "The PDF is too large or complex to read."

# Field flags (ISO 32000-1, tables 221, 226, 228 and 230).
FF_READ_ONLY = 1 << 0
FF_REQUIRED = 1 << 1
FF_MULTILINE = 1 << 12
FF_PASSWORD = 1 << 13
FF_RADIO = 1 << 15
FF_PUSHBUTTON = 1 << 16
FF_COMBO = 1 << 17
FF_EDIT = 1 << 18
FF_FILE_SELECT = 1 << 20
FF_MULTI_SELECT = 1 << 21
FF_COMB = 1 << 24

# Annotation flags.
ANNOT_HIDDEN = 1 << 1
ANNOT_NO_VIEW = 1 << 5

# Permission bits (/P): "modify annotations / fill fields" and "fill existing fields".
PERM_MODIFY = 1 << 3
PERM_ANNOTATE = 1 << 5
PERM_FILL_FIELDS = 1 << 8

_INHERITABLE = ("/FT", "/Ff", "/V", "/MaxLen", "/Opt", "/DA")


@dataclass
class LoadedForm:
    """A parsed form. `field_paths` locate each field in the AcroForm tree, so
    the filler can find the same objects in a cloned writer."""

    info: FormInfo
    reader: PdfReader  # already decrypted
    field_paths: dict[str, tuple[int, ...]]
    source_bytes: bytes = field(repr=False)

    def plain_pdf(self) -> bytes:
        """The PDF bytes, decrypted if the original was encrypted.

        Raises InvalidInputError if the decrypted copy can't be written.
        """
        if not self.info.encrypted:
            return self.source_bytes
        try:
            with pypdf_limits():  # the copy loads objects that open_form never read
                buf = io.BytesIO()
                PdfWriter(clone_from=self.reader).write(buf)
                return buf.getvalue()
        except LimitReachedError as exc:
            log.info("PDF copy hit a limit: %s", type(exc).__name__)
            raise InvalidInputError(TOO_COMPLEX) from exc
        except Exception as exc:  # pypdf raises many types on damaged files
            log.info("PDF copy failed: %s", type(exc).__name__)
            raise InvalidInputError("The PDF could not be read. It may be damaged.") from exc


def open_form(data: bytes, *, limits: Limits = Limits(), for_filling: bool = True) -> LoadedForm:
    """Validates and inspects a PDF.

    With `for_filling=False` (reference forms and PDF documents), a PDF without
    fields or with restrictive permissions is accepted.
    """
    with pypdf_limits():  # pypdf decodes lazily, both while parsing and while inspecting
        return _open_form(data, limits, for_filling)


def _open_form(data: bytes, limits: Limits, for_filling: bool) -> LoadedForm:
    if len(data) > limits.max_file_bytes:
        raise InvalidInputError(f"The PDF is too large (limit {limits.max_file_bytes // MB} MB).")
    if PDF_MAGIC not in data[:1024]:
        raise InvalidInputError("The file is not a PDF.")

    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        encrypted = reader.is_encrypted
        permissions = _decrypt(reader) if encrypted else None
        page_count = _page_count(reader)
    except (UnsupportedFormError, InvalidInputError):
        raise
    except LimitReachedError as exc:  # an oversized stream or page tree
        log.info("PDF parse hit a limit: %s", type(exc).__name__)
        raise InvalidInputError(TOO_COMPLEX) from exc
    except Exception as exc:  # pypdf raises many types on malformed input
        log.info("PDF parse failed: %s", type(exc).__name__)
        raise InvalidInputError("The PDF could not be read. It may be damaged.") from exc

    if page_count > limits.max_pages:
        raise InvalidInputError(f"The PDF has too many pages (limit {limits.max_pages}).")

    try:
        inspected = _inspect(reader, limits)
    except RecursionError as exc:
        raise InvalidInputError("The PDF's form structure is too deeply nested.") from exc
    except (UnsupportedFormError, InvalidInputError):
        raise
    except LimitReachedError as exc:
        log.info("form inspection hit a limit: %s", type(exc).__name__)
        raise InvalidInputError(TOO_COMPLEX) from exc
    except Exception as exc:
        log.info("form inspection failed: %s", type(exc).__name__)
        raise InvalidInputError("The PDF's form fields could not be read. The file may be damaged.") from exc
    fields, paths, warnings, has_xfa, needs_rendering = inspected

    if for_filling:
        if not fields:
            if has_xfa or needs_rendering:
                raise UnsupportedFormError(
                    "This is an XFA form, a format that only Adobe software fully supports. "
                    "PDF Wizard can fill standard (AcroForm) PDF forms."
                )
            raise UnsupportedFormError("This PDF has no fillable form fields.")
        if permissions is not None and not permissions & (PERM_ANNOTATE | PERM_FILL_FIELDS):
            raise UnsupportedFormError("This PDF's security settings don't allow filling in form fields.")
        if has_xfa:
            warnings.append(
                "This form also contains XFA data. It will be removed so the filled values "
                "display the same in every PDF viewer."
            )

    info = FormInfo(
        sha256=hashlib.sha256(data).hexdigest(),
        page_count=page_count,
        fields=fields,
        had_xfa=has_xfa,
        encrypted=encrypted,
        permissions=permissions,
        warnings=warnings,
    )
    return LoadedForm(info=info, reader=reader, field_paths=paths, source_bytes=data)


def _decrypt(reader: PdfReader) -> int | None:
    """Opens PDFs that have an empty user password. Returns the /P flags, or
    None when the empty password is also the owner password (no restrictions)."""
    result = reader.decrypt("")
    if result == PasswordType.NOT_DECRYPTED:
        raise UnsupportedFormError("This PDF is password-protected. Remove the password and try again.")
    if result == PasswordType.OWNER_PASSWORD:
        return None
    perms = reader.user_access_permissions
    return None if perms is None else int(perms)


def _page_count(reader: PdfReader) -> int:
    """The number of pages pypdf will visit.

    For encrypted files pypdf returns /Pages /Count as declared, so count the
    page tree's leaves instead (pypdf caps the tree's size and depth). The real
    count is stored back, because later page iteration and copies use /Count.
    """
    if not reader.is_encrypted:
        return len(reader.pages)
    try:
        reader.get_page(0)  # flattens the page tree
    except IndexError:  # no pages
        pass
    count = len(reader.flattened_pages or ())
    pages = _get(reader.root_object, "/Pages")
    if isinstance(pages, DictionaryObject) and pages.get("/Count") != count:
        log.info("encrypted PDF declares a wrong page count; using the real count (%d)", count)
        pages[NameObject("/Count")] = NumberObject(count)
    return count


# --------------------------------------------------------------------------- #
# Field tree
# --------------------------------------------------------------------------- #


def _get(d: DictionaryObject, key: str, default: Any = None) -> Any:
    """dict.get that resolves indirect references (pypdf's get() doesn't)."""
    value = d.get(key)
    return default if value is None else value.get_object()


def _key(ref: Any, obj: Any) -> int:
    return ref.idnum if isinstance(ref, IndirectObject) else id(obj)


@dataclass
class _RawField:
    node: DictionaryObject
    path: tuple[int, ...]
    name: str
    attrs: dict[str, Any]  # inherited attributes, resolved
    widgets: list[tuple[int, DictionaryObject]]  # (identity key, widget dict)


def _walk(fields_array: Any, limits: Limits, warnings: list[str]) -> list[_RawField]:
    found: list[_RawField] = []
    visited: set[int] = set()
    truncated = False

    def visit(ref: Any, path: tuple[int, ...], names: list[str], inherited: dict[str, Any]) -> None:
        nonlocal truncated
        if len(found) >= limits.max_fields:
            truncated = True
            return
        node = ref.get_object() if ref is not None else None
        if not isinstance(node, DictionaryObject) or len(path) > MAX_TREE_DEPTH:
            return
        key = _key(ref, node)
        if key in visited:
            return  # cycle or shared node
        visited.add(key)

        attrs = dict(inherited)
        for name in _INHERITABLE:
            if name in node:
                attrs[name] = _get(node, name)
        partial = node.get("/T")
        qualified = [*names, str(partial)] if partial is not None else names

        kids = _get(node, "/Kids")
        kid_refs = list(kids) if isinstance(kids, ArrayObject) else []
        field_kids = [(i, k) for i, k in enumerate(kid_refs) if _is_field(k)]
        if field_kids:
            for i, kid in field_kids:
                visit(kid, (*path, i), qualified, attrs)
            return

        widgets: list[tuple[int, DictionaryObject]] = []
        if kid_refs:
            for kid in kid_refs:
                widget = kid.get_object()
                if isinstance(widget, DictionaryObject):
                    widgets.append((_key(kid, widget), widget))
        elif node.get("/Subtype") == "/Widget":
            widgets.append((key, node))
        found.append(_RawField(node=node, path=path, name=".".join(qualified), attrs=attrs, widgets=widgets))

    for i, ref in enumerate(list(fields_array or [])):
        visit(ref, (i,), [], {})
    if truncated:
        warnings.append(f"This form has more than {limits.max_fields} fields; only the first {limits.max_fields} are used.")
    return found


def _is_field(ref: Any) -> bool:
    kid = ref.get_object() if ref is not None else None
    return isinstance(kid, DictionaryObject) and "/T" in kid


def _inspect(reader: PdfReader, limits: Limits):
    root = reader.root_object
    acro = _get(root, "/AcroForm")
    warnings: list[str] = []
    has_xfa = isinstance(acro, DictionaryObject) and "/XFA" in acro
    needs_rendering = bool(_get(root, "/NeedsRendering", False))
    if not isinstance(acro, DictionaryObject):
        return [], {}, warnings, has_xfa, needs_rendering

    raw_fields = _walk(_get(acro, "/Fields"), limits, warnings)
    page_of = _annotation_pages(reader)
    texts = _PageTexts(reader)
    xfa_labels = _xfa_labels(acro) if has_xfa else {}

    fields: list[FormField] = []
    paths: dict[str, tuple[int, ...]] = {}
    signed = False
    for raw in raw_fields:
        ftype = _field_type(raw.attrs, raw.widgets)
        if ftype is None:
            continue
        field_id = f"F{len(fields) + 1}"
        built = _build_field(field_id, raw, ftype, page_of, texts, xfa_labels.get(raw.name))
        fields.append(built)
        paths[field_id] = raw.path
        if ftype is FieldType.SIGNATURE and raw.attrs.get("/V") is not None:
            signed = True
    if signed:
        warnings.append("This form has a signed signature field. Filling the form invalidates the signature.")
    return fields, paths, warnings, has_xfa, needs_rendering


def _field_type(attrs: dict[str, Any], widgets: list[tuple[int, DictionaryObject]] | tuple[Any, ...] = ()) -> FieldType | None:
    ft = attrs.get("/FT")
    flags = _int(attrs.get("/Ff"))
    if ft == "/Tx":
        return FieldType.TEXT
    if ft == "/Btn":
        if flags & FF_PUSHBUTTON:
            return FieldType.BUTTON
        if flags & FF_RADIO:
            return FieldType.RADIO
        if len(widgets) > 1 and len({s for _, w in widgets for s in _states(w)}) > 1:
            return FieldType.RADIO
        return FieldType.CHECKBOX
    if ft == "/Ch":
        return FieldType.CHOICE
    if ft == "/Sig":
        return FieldType.SIGNATURE
    return None


def _build_field(field_id: str, raw: _RawField, ftype: FieldType, page_of: dict[int, int],
                 texts: _PageTexts, xfa_label: str | None = None) -> FormField:
    attrs = raw.attrs
    flags = _int(attrs.get("/Ff"))
    widgets = raw.widgets
    first_page = first_rect = None
    visible = False
    boxes: list[tuple[int, tuple[float, float, float, float]]] = []
    for key, widget in widgets:
        rect = _rect(widget)
        page = page_of.get(key)
        if page is None:
            page = _page_from_p(widget, page_of)
        if first_page is None and page is not None:
            first_page, first_rect = page, rect
        if page is not None and rect is not None:
            boxes.append((page, rect))
        flags_annot = _int(_get(widget, "/F"))
        has_area = rect is not None and rect[2] - rect[0] > 0.5 and rect[3] - rect[1] > 0.5
        if not flags_annot & (ANNOT_HIDDEN | ANNOT_NO_VIEW) and has_area and page is not None:
            visible = True

    raw_tu = _text(_get(raw.node, "/TU"))
    if legacy_heuristics_enabled():
        if xfa_label and (not raw_tu or len(raw_tu.strip()) > MAX_LABEL_CHARS or "I R S" in raw_tu):
            label = xfa_label
        else:
            label = _condense_long_tooltip(raw_tu) if raw_tu else xfa_label
    elif xfa_label and (not raw_tu or len(raw_tu.strip()) > MAX_LABEL_CHARS):
        label = xfa_label
    else:
        label = raw_tu or xfa_label
    options = _options(ftype, attrs, widgets, page_of, texts, field_label=label if len(widgets) == 1 else None)
    if not label and first_page is not None and first_rect is not None:
        if ftype is FieldType.RADIO:
            label = texts.group_label_for(
                first_page, [r for p, r in boxes if p == first_page], exclude={o.label for o in options if o.label}
            )
        else:
            label = texts.label_for(first_page, first_rect, prefer_right=ftype is FieldType.CHECKBOX)

    max_len = _int(attrs.get("/MaxLen")) if ftype is FieldType.TEXT else 0
    return FormField(
        id=field_id,
        name=raw.name or field_id,
        label=_clean_label(label),
        type=ftype,
        page=first_page,
        rect=first_rect,
        options=options,
        max_len=max_len if max_len > 0 else None,
        required=bool(flags & FF_REQUIRED),
        read_only=bool(flags & FF_READ_ONLY),
        multiline=ftype is FieldType.TEXT and bool(flags & FF_MULTILINE),
        comb=ftype is FieldType.TEXT and bool(flags & FF_COMB) and max_len > 0,
        editable=ftype is FieldType.CHOICE and bool(flags & FF_COMBO) and bool(flags & FF_EDIT),
        multi_select=ftype is FieldType.CHOICE and bool(flags & FF_MULTI_SELECT),
        hidden=not visible,
        password=ftype is FieldType.TEXT and bool(flags & FF_PASSWORD),
        file_select=ftype is FieldType.TEXT and bool(flags & FF_FILE_SELECT),
        format_hint=_format_hint(raw.node, widgets) if ftype is FieldType.TEXT else None,
        current_value=_current_value(ftype, attrs, widgets),
    )


def _options(ftype: FieldType, attrs: dict[str, Any], widgets, page_of, texts: _PageTexts,
             *, field_label: str | None = None) -> list[Option]:
    options: list[Option] = []
    if ftype in (FieldType.CHECKBOX, FieldType.RADIO):
        opt = attrs.get("/Opt")
        export_values = opt if isinstance(opt, ArrayObject) else ()  # indexed, not copied: it may be shared
        seen: set[str] = set()
        for index, (key, widget) in enumerate(widgets):
            if len(options) >= MAX_OPTIONS:
                break
            states = _states(widget)
            new = [state for state in states if state not in seen][: MAX_OPTIONS - len(options)]
            if not new:
                continue
            seen.update(new)
            # All states of a widget share its label, so look it up once. A widget
            # with dozens of on-states is no real checkbox: skip the page-text search.
            label = _widget_label(index, key, widget, export_values, page_of, texts,
                                  fallback=field_label, search_page=len(states) <= MAX_LABELED_STATES)
            options.extend(Option(value=state, label=label if label != state else None) for state in new)
    elif ftype is FieldType.CHOICE:
        opt = attrs.get("/Opt")
        for entry in opt[:MAX_OPTIONS] if isinstance(opt, ArrayObject) else []:
            entry = entry.get_object()
            if isinstance(entry, ArrayObject) and len(entry) >= 2:
                value, label = _text(entry[0].get_object()), _text(entry[1].get_object())
            else:
                value, label = _text(entry), None
            if value and value.strip():
                options.append(Option(value=value, label=label if label and label != value else None))
    return options


def _widget_label(index: int, key: int, widget: DictionaryObject, export_values, page_of, texts: _PageTexts,
                  *, fallback: str | None = None, search_page: bool) -> str | None:
    """A checkbox or radio widget's label: its /Opt entry, field label, else the page text next to it."""
    label = _text(export_values[index].get_object()) if index < len(export_values) else fallback
    if not label and search_page:
        page, rect = page_of.get(key), _rect(widget)
        if page is not None and rect is not None:
            label = texts.label_for(page, rect, prefer_right=True)
    return _clean_label(label)


def _states(widget: DictionaryObject) -> list[str]:
    ap = _get(widget, "/AP")
    normal = _get(ap, "/N") if isinstance(ap, DictionaryObject) else None
    if not isinstance(normal, DictionaryObject) or isinstance(normal, StreamObject):
        return []
    # A field keeps at most MAX_OPTIONS options, so later names are never used.
    return list(itertools.islice((str(k)[1:] for k in normal.keys() if str(k) != "/Off"), MAX_OPTIONS))


def _current_value(ftype: FieldType, attrs: dict[str, Any], widgets) -> str | None:
    value = attrs.get("/V")
    if ftype in (FieldType.CHECKBOX, FieldType.RADIO):
        if value is None and ftype is FieldType.CHECKBOX and widgets:
            value = _get(widgets[0][1], "/AS")
        text = str(value)[1:] if value is not None and str(value).startswith("/") else _text(value)
        return text or None
    if isinstance(value, ArrayObject):
        value = value[0].get_object() if value else None
    return _text(value) or None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    # Streams (allowed for /V, but rare) are not decoded: a few KB can inflate to
    # many MB, which pypdf then keeps cached as long as the reader lives.
    if isinstance(value, (StreamObject, ArrayObject, DictionaryObject)):
        return None
    return str(value)


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _rect(widget: DictionaryObject) -> tuple[float, float, float, float] | None:
    rect = _get(widget, "/Rect")
    try:
        x1, y1, x2, y2 = (float(v.get_object()) for v in rect)
    except Exception:
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def _annotation_pages(reader: PdfReader) -> dict[int, int]:
    """Maps annotation identity -> 1-based page number."""
    pages: dict[int, int] = {}
    read: set[int] = set()  # pages can share one /Annots array; reading it again changes nothing
    for number, page in enumerate(reader.pages, start=1):
        annots = _get(page, "/Annots")
        if isinstance(annots, ArrayObject) and id(annots) not in read:
            read.add(id(annots))
            for ref in annots:
                pages.setdefault(_key(ref, ref.get_object()), number)
        pages.setdefault(_key(page.indirect_reference, page), number)
    return pages


def _page_from_p(widget: DictionaryObject, page_of: dict[int, int]) -> int | None:
    ref = widget.get("/P")
    if isinstance(ref, IndirectObject):
        return page_of.get(ref.idnum)
    return None


# --------------------------------------------------------------------------- #
# Format hints (parsed, never executed)
# --------------------------------------------------------------------------- #

# AFDate_Format(n) / AFDate_Keystroke(n) index table from Acrobat's AForm.js.
_AF_DATE_FORMATS = (
    "m/d", "m/d/yy", "mm/dd/yy", "mm/yy", "d-mmm", "d-mmm-yy", "dd-mmm-yy", "yy-mm-dd",
    "mmm-yy", "mmmm-yy", "mmm d, yyyy", "mmmm d, yyyy", "m/d/yy h:MM tt", "m/d/yy HH:MM",
)
_SPECIAL = {0: "zip", 1: "zip4", 2: "phone", 3: "ssn"}
_STR = r"""\(\s*["']([^"'\\]{1,64})["']"""
_RE_DATE_EX = re.compile(r"AFDate_(?:Format|Keystroke)Ex\s*" + _STR)
_RE_DATE = re.compile(r"AFDate_(?:Format|Keystroke)\s*\(\s*(\d+)\s*\)")
_RE_SPECIAL_EX = re.compile(r"AFSpecial_(?:Format|Keystroke)Ex\s*" + _STR)
_RE_SPECIAL = re.compile(r"AFSpecial_(?:Format|Keystroke)\s*\(\s*(\d)\s*\)")
_RE_NUMBER = re.compile(r"AFNumber_(?:Format|Keystroke)\s*\(\s*(\d+)\s*,\s*(\d+)")
_RE_PERCENT = re.compile(r"AFPercent_(?:Format|Keystroke)\s*\(\s*(\d+)\s*,\s*(\d+)")


def _format_hint(node: DictionaryObject, widgets) -> str | None:
    holders = [node] + [w for _, w in widgets if w is not node]
    for event in ("/F", "/K"):  # prefer the format script over the keystroke script
        for holder in holders:
            aa = _get(holder, "/AA")
            if isinstance(aa, DictionaryObject):
                hint = parse_format_script(_script(_get(aa, event)))
                if hint:
                    return hint
    return None


def _script(action: Any) -> str:
    if not isinstance(action, DictionaryObject) or _get(action, "/S") != "/JavaScript":
        return ""
    js = _get(action, "/JS")
    if isinstance(js, StreamObject):
        return _stream_script(js)
    return str(js)[:MAX_SCRIPT_CHARS] if js is not None else ""


def _stream_script(stream: StreamObject) -> str:
    """A /JS stream's text, inflating at most MAX_SCRIPT_CHARS + 1 bytes.

    pypdf's get_data() would inflate the whole stream and cache the result. A
    script that is longer than MAX_SCRIPT_CHARS, or encoded other than plain or
    Flate, counts as no script: Acrobat's format calls are short.
    """
    filters = _get(stream, "/Filter")
    if isinstance(filters, ArrayObject) and len(filters) == 1:
        filters = filters[0].get_object()
    raw = stream._data  # the encoded bytes (already decrypted); pypdf has no public accessor
    try:
        if filters is None:
            data = bytes(raw[: MAX_SCRIPT_CHARS + 1])
        elif filters in ("/FlateDecode", "/Fl"):
            data = zlib.decompressobj().decompress(raw, MAX_SCRIPT_CHARS + 1)
        else:
            return ""
    except (zlib.error, TypeError):
        return ""
    return data.decode("latin-1") if len(data) <= MAX_SCRIPT_CHARS else ""


def parse_format_script(js: str) -> str | None:
    """Recognizes Acrobat's built-in AF* format functions in a script."""
    if not js:
        return None
    if m := _RE_DATE_EX.search(js):
        return f"date:{m.group(1).strip()}"
    if (m := _RE_DATE.search(js)) and int(m.group(1)) < len(_AF_DATE_FORMATS):
        return f"date:{_AF_DATE_FORMATS[int(m.group(1))]}"
    if m := _RE_SPECIAL_EX.search(js):
        return f"mask:{m.group(1)}"
    if (m := _RE_SPECIAL.search(js)) and int(m.group(1)) in _SPECIAL:
        return _SPECIAL[int(m.group(1))]
    if m := _RE_PERCENT.search(js):
        return f"percent:{int(m.group(1))}:{int(m.group(2))}"
    if m := _RE_NUMBER.search(js):
        return f"number:{int(m.group(1))}:{int(m.group(2))}"
    return None


# --------------------------------------------------------------------------- #
# Labels from nearby page text
# --------------------------------------------------------------------------- #


@dataclass
class _Line:
    x: float
    y: float  # baseline
    size: float
    text: str

    @property
    def width(self) -> float:  # estimated; pypdf doesn't report glyph widths here
        return len(self.text) * self.size * 0.45

    @property
    def x2(self) -> float:
        return self.x + self.width


@dataclass
class _PageLines:
    """A page's text lines in reading order, indexed by baseline."""

    lines: list[_Line]
    keys: list[int] = field(init=False)  # -round(y): non-decreasing, since _merge_runs sorts by it
    max_size: float = field(init=False)

    def __post_init__(self) -> None:
        self.keys = [-round(line.y) for line in self.lines]
        self.max_size = max((line.size for line in self.lines), default=1.0)

    def between(self, y_lo: float, y_hi: float) -> list[_Line]:
        """The lines with a baseline in [y_lo, y_hi] (and a few within rounding of it), in reading order."""
        lo = bisect.bisect_left(self.keys, -math.ceil(y_hi)) if math.isfinite(y_hi) else 0
        hi = bisect.bisect_right(self.keys, -math.floor(y_lo)) if math.isfinite(y_lo) else len(self.keys)
        return self.lines[lo:hi]


class _PageTexts:
    """Lazily extracts positioned text lines per page for label inference.

    The work is bounded: a search scores only the lines whose baseline is close
    enough to the box to qualify, a page keeps at most MAX_LABEL_LINES lines, and
    a form runs at most MAX_LABEL_LOOKUPS searches.
    """

    def __init__(self, reader: PdfReader):
        self._reader = reader
        self._cache: dict[int, _PageLines] = {}
        self._found: dict[tuple[Any, ...], str | None] = {}
        self._lookups = 0

    def lines(self, page_number: int) -> _PageLines:
        if page_number not in self._cache:
            self._cache[page_number] = _PageLines(self._extract(page_number))
        return self._cache[page_number]

    def _extract(self, page_number: int) -> list[_Line]:
        runs: list[_Line] = []

        def visitor(text, cm, tm, _font, font_size):
            text = " ".join(str(text).split())
            if not text or len(runs) > 20_000:
                return
            try:
                x = tm[4] * cm[0] + tm[5] * cm[2] + cm[4]
                y = tm[4] * cm[1] + tm[5] * cm[3] + cm[5]
                size = float(font_size) * math.hypot(tm[0], tm[1]) * math.hypot(cm[0], cm[1])
            except (TypeError, ValueError, IndexError):
                return
            if all(math.isfinite(v) for v in (x, y, size)):
                runs.append(_Line(float(x), float(y), max(size, 1.0), text))

        try:
            page_text(self._reader.pages[page_number - 1], visitor_text=visitor)
        except Exception as exc:  # label inference is best effort
            log.debug("text extraction failed on a page: %s", type(exc).__name__)
            return []
        return _merge_runs(runs)[:MAX_LABEL_LINES]

    def _spend(self) -> bool:
        """Counts one search; False once the form's budget is used up."""
        if self._lookups < MAX_LABEL_LOOKUPS:
            self._lookups += 1
            return True
        if self._lookups == MAX_LABEL_LOOKUPS:  # log the first refusal only
            self._lookups += 1
            log.info("label inference stopped after %d searches", MAX_LABEL_LOOKUPS)
        return False

    def label_for(self, page_number: int, rect, *, prefer_right: bool) -> str | None:
        key = (page_number, tuple(rect), prefer_right)
        if key in self._found:  # a checkbox's field label repeats its option's search
            return self._found[key]
        if not self._spend():
            return None
        x1, y1, x2, y2 = rect
        page = self.lines(page_number)
        size = page.max_size  # _label_score accepts lines up to 3 sizes above or 2 below the box
        best: tuple[float, str] | None = None
        for line in page.between(y1 - 2 * size - 10, y2 + 3 * size + 12):
            score = _label_score(line, x1, y1, x2, y2, prefer_right)
            if score is not None and (best is None or score < best[0]):
                best = (score, line.text)
        self._found[key] = label = best[1] if best else None
        return label

    def group_label_for(self, page_number: int, rects, *, exclude: set[str]) -> str | None:
        """A radio group's label: the text printed left of its top-left button.

        Text equal to an option label is skipped, for layouts that print each
        option's label left of its button ("Yes ( ) No ( )").
        """
        if not rects:
            return None
        top = max(r[3] for r in rects)
        x1, y1, x2, y2 = min((r for r in rects if r[3] >= top - 3), key=lambda r: r[0])
        if not self._spend():
            return None
        candidates = sorted(
            (max(0.0, x1 - line.x2), line.text)
            for line in self.lines(page_number).between(y1 - 3, y2 + 3)  # the same row
            if _is_left_label(line, x1, y1, x2, y2)
        )
        return next((text for _, text in candidates if _clean_label(text) not in exclude), None)


def _merge_runs(runs: list[_Line]) -> list[_Line]:
    runs.sort(key=lambda r: (-round(r.y), r.x))
    lines: list[_Line] = []
    for run in runs:
        last = lines[-1] if lines else None
        max_gap = 3.0 * max(last.size, run.size) if last and len(last.text) <= 3 and last.text[0].isdigit() else (1.5 * run.size if last else 0.0)
        if last and abs(last.y - run.y) < 2 and -2 <= run.x - last.x2 < max_gap:
            gap = " " if run.x - last.x2 > 0.15 * run.size else ""
            lines[-1] = _Line(last.x, last.y, max(last.size, run.size), last.text + gap + run.text)
        else:
            lines.append(run)
    return lines


def _is_left_label(line: _Line, x1: float, y1: float, x2: float, y2: float) -> bool:
    """Same row, ending at (or, given the width estimate, slightly past) the box's left edge."""
    same_row = y1 - 3 <= line.y <= y2 + 3
    return same_row and line.x < x1 and line.x2 <= x1 + max(8.0, 0.3 * line.width) and x1 - line.x2 < 250


def _label_score(line: _Line, x1: float, y1: float, x2: float, y2: float, prefer_right: bool) -> float | None:
    if not any(ch.isalnum() for ch in line.text):
        return None
    penalty = 0.0 if sum(ch.isalpha() for ch in line.text) >= 2 else 25.0
    same_row = y1 - 3 <= line.y <= y2 + 3
    overlaps = line.x < x2 and line.x2 > x1 - 20
    if same_row and x1 - 1 <= line.x < x2 and line.size < (y2 - y1):
        return penalty  # printed inside the box
    if _is_left_label(line, x1, y1, x2, y2):
        return max(0.0, x1 - line.x2) + (40 if prefer_right else 0) + penalty
    if same_row and line.x >= x2 - 2 and line.x - x2 < 150:
        return (line.x - x2) + (0 if prefer_right else 120) + penalty
    if overlaps and y2 < line.y <= y2 + 3 * line.size + 12:
        return (line.y - y2) * 1.5 + abs(line.x - x1) * 0.05 + 10 + penalty
    if overlaps and y1 - 2 * line.size - 10 <= line.y < y1:
        return (y1 - line.y) * 2 + abs(line.x - x1) * 0.05 + 30 + penalty
    return None


# --------------------------------------------------------------------------- #
# Labels from XFA template (hybrid AcroForm + XFA forms)
# --------------------------------------------------------------------------- #

_SPACED_CAPS = re.compile(r"\b[A-Z](?: [A-Z])+\b")
_ITEM_MARKER = re.compile(r"(?:^|[.:;]\s+(?:or\s+)?)((?:\d+[a-z]?(?:\([a-z0-9]+\))?|\([a-z]\))\.?\s+)")


def _xfa_labels(acro: DictionaryObject) -> dict[str, str]:
    """Extracts field labels from an /AcroForm /XFA template stream, keyed by full AcroForm name."""
    xml_text = _xfa_template(_get(acro, "/XFA"))
    if not xml_text or "<!entity" in xml_text.lower():
        return {}
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return {}
    labels: dict[str, str] = {}

    def walk(elem: ET.Element, path: str, depth: int) -> None:
        if depth > MAX_TREE_DEPTH:
            return
        counts: dict[str, int] = defaultdict(int)
        for child in elem:
            tag = child.tag.split("}")[-1]
            if tag in ("subform", "field", "exclGroup", "area"):
                name = child.attrib.get("name")
                if name:
                    idx = counts[name]
                    counts[name] += 1
                    cur_path = f"{path}.{name}[{idx}]" if path else f"{name}[{idx}]"
                else:
                    cur_path = path
                if tag == "field":
                    if label := _xfa_field_label(child):
                        labels[cur_path] = label
                else:
                    walk(child, cur_path, depth + 1)
            else:
                walk(child, path, depth + 1)

    walk(root, "", 0)
    return labels


def _xfa_template(xfa: Any) -> str:
    stream: StreamObject | None = None
    if isinstance(xfa, StreamObject):
        stream = xfa
    elif isinstance(xfa, ArrayObject):
        for i in range(0, len(xfa) - 1, 2):
            if str(xfa[i].get_object()) == "template":
                candidate = xfa[i + 1].get_object()
                if isinstance(candidate, StreamObject):
                    stream = candidate
                break
    if stream is None:
        return ""
    filters = _get(stream, "/Filter")
    if isinstance(filters, ArrayObject) and len(filters) == 1:
        filters = filters[0].get_object()
    raw = stream._data
    try:
        if filters is None:
            data = bytes(raw[: MAX_XFA_BYTES + 1])
        elif filters in ("/FlateDecode", "/Fl"):
            data = zlib.decompressobj().decompress(raw, MAX_XFA_BYTES + 1)
        else:
            return ""
    except (zlib.error, TypeError):
        return ""
    return data.decode("utf-8", errors="ignore") if len(data) <= MAX_XFA_BYTES else ""


def _xfa_field_label(field_elem: ET.Element) -> str | None:
    caption = tooltip = speak = None
    for child in field_elem:
        tag = child.tag.split("}")[-1]
        if tag == "assist":
            for ac in child:
                actag = ac.tag.split("}")[-1]
                if actag == "speak":
                    speak = " ".join(" ".join(ac.itertext()).split())
                elif actag == "toolTip":
                    tooltip = " ".join(" ".join(ac.itertext()).split())
        elif tag == "caption":
            caption = " ".join(" ".join(child.itertext()).split())
    if tooltip and len(tooltip.strip()) >= 2:
        return tooltip.strip()
    cap = caption.strip(" .:$▶▲") if caption else ""
    spk = _clean_speak(speak) if speak else ""
    has_good_cap = (
        sum(ch.isalnum() for ch in cap) >= 4
        and cap.lower() not in ("code (if any)",)
        and not ("(" in spk and "(" not in cap)
    )
    if not spk:
        return cap or None
    matches = list(_ITEM_MARKER.finditer(spk))
    if has_good_cap:
        if matches:
            num = matches[-1].group(1).strip().rstrip(".")
            if not (cap.startswith(num) or cap[0] in "(0123456789"):
                return f"{num}. {cap}"[:MAX_LABEL_CHARS]
        return cap[:MAX_LABEL_CHARS]
    if len(spk) <= MAX_LABEL_CHARS:
        return spk
    if matches:
        return spk[matches[-1].start(1) :][:MAX_LABEL_CHARS]
    sentences = [p.strip() for p in spk.split(". ") if p.strip()]
    tail = sentences[-1]
    for s in reversed(sentences[:-1]):
        cand = f"{s}. {tail}"
        if len(cand) <= MAX_LABEL_CHARS:
            tail = cand
        else:
            break
    return tail[:MAX_LABEL_CHARS]


def _clean_speak(text: str) -> str:
    text = _SPACED_CAPS.sub(lambda m: m.group(0).replace(" ", ""), text)
    text = text.replace("P. O.", "P.O.").replace("U. S.", "U.S.")
    return re.sub(r"\s*\$?\.?\s*$", "", text).strip()


def _condense_long_tooltip(text: str) -> str:
    """Condenses oversized /TU tooltips that contain multi-sentence instructions."""
    cleaned = _clean_speak(" ".join(text.split()))
    if len(cleaned) <= MAX_LABEL_CHARS:
        return cleaned
    matches = list(_ITEM_MARKER.finditer(cleaned))
    if matches:
        return cleaned[matches[-1].start(1) :][:MAX_LABEL_CHARS]
    return cleaned[:MAX_LABEL_CHARS]


_LEADERS_START = re.compile(r"^(?:[\u2018\u2019\u2022\u25a0-\u25ff\u2610-\u2612]\s+)+")
_LEADERS = re.compile(r"[\s.:_\u2026]+$")


def _clean_label(label: str | None) -> str | None:
    if not label:
        return None
    label = "".join(ch for ch in label if ch.isprintable())
    label = _LEADERS_START.sub("", " ".join(label.split()))
    label = _LEADERS.sub("", label)
    return label[:MAX_LABEL_CHARS] or None
