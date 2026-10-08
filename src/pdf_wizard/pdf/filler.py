"""Filler: writes checked values into a copy of the form (stage 6).

pypdf matches fields by name, and partial names can collide (a top-level
"Name" also matches "Spouse.Name"). To write exactly one field, the filler
locates each field by its path in the field tree, gives it a temporary unique
mapping name (/TM, which pypdf checks first), and fills by that name.

Flattening is done here rather than with pypdf's `flatten=True`, which reuses
one XObject name per field (wrong for radio groups) and ignores the
appearance's /BBox and /Matrix.
"""

from __future__ import annotations

import io
import logging
import math
import re
import secrets
from dataclasses import dataclass, field
from typing import Any

from pypdf import PdfWriter
from pypdf.constants import UserAccessPermissions
from pypdf.errors import LimitReachedError
from pypdf.generic import (
    ArrayObject,
    DecodedStreamObject,
    DictionaryObject,
    IndirectObject,
    NameObject,
    NumberObject,
    StreamObject,
    TextStringObject,
)

from ..errors import PdfWizardError, UnsupportedFormError
from ..models import FieldType, FormField
from .inspector import ANNOT_HIDDEN, ANNOT_NO_VIEW, FF_COMBO, PERM_MODIFY, LoadedForm, _get
from .limits import pypdf_limits

log = logging.getLogger(__name__)

_MISSING = object()
_BUTTONS = (FieldType.CHECKBOX, FieldType.RADIO)


@dataclass
class FilledPdf:
    pdf: bytes = field(repr=False)
    failed: dict[str, str] = field(default_factory=dict)  # field id -> safe detail
    warnings: list[str] = field(default_factory=list)


def fill_pdf(form: LoadedForm, values: dict[str, str], *, flatten: bool = False) -> FilledPdf:
    """Writes `values` (field id -> PDF-ready value) and returns the new PDF.

    Text/choice values are written as text; checkbox/radio values are state
    names without "/" ("Yes", "Off"). Fields that can't be written are left
    unchanged and listed in `failed`.
    """
    info = form.info
    if flatten and info.permissions is not None and not info.permissions & PERM_MODIFY:
        raise UnsupportedFormError(
            "This PDF's security settings don't allow flattening. Fill it without flattening."
        )
    try:
        with pypdf_limits():  # cloning, drawing and writing decode streams lazily
            return _fill_pdf(form, values, flatten=flatten)
    except PdfWizardError:
        raise
    except LimitReachedError as exc:
        log.info("filling hit a limit: %s", type(exc).__name__)
        raise UnsupportedFormError("This PDF couldn't be filled because part of it is too large or complex.") from exc
    except Exception as exc:  # damaged objects outside any one field (pypdf raises many types)
        log.info("filling failed: %s", type(exc).__name__)
        raise UnsupportedFormError("This PDF couldn't be filled because part of it is damaged.") from exc


def _fill_pdf(form: LoadedForm, values: dict[str, str], *, flatten: bool) -> FilledPdf:
    info = form.info
    writer = PdfWriter(clone_from=form.reader)
    acro = _get(writer.root_object, "/AcroForm")
    warnings: list[str] = []
    _strip_incompatible(writer.root_object, acro, warnings)

    ctx = _Context(writer, acro)
    if info.had_xfa:
        for fld in info.fields:
            if fld.label and (loc := ctx.locate(form.field_paths.get(fld.id))) is not None and "/TU" not in loc.node:
                loc.node[NameObject("/TU")] = TextStringObject(fld.label)
    fields = {f.id: f for f in info.fields}
    failed: dict[str, str] = {}
    for field_id, value in values.items():
        fld = fields.get(field_id)
        located = ctx.locate(form.field_paths.get(field_id))
        if fld is None or located is None or not fld.writable:
            failed[field_id] = "The field could not be found in the PDF."
            continue
        try:
            _write(ctx, fld, located, value, allow_fallback=not flatten)
            for identity, _ in located.widgets:
                ctx.widget_field[identity] = field_id
        except Exception as exc:  # pypdf raises many types on unusual forms
            log.info("could not write a field: %s", type(exc).__name__)
            failed[field_id] = "The value could not be written into this field."

    if ctx.fallbacks:
        warnings.append(
            f"{ctx.fallbacks} field(s) were filled without a pre-drawn appearance; "
            "some PDF viewers show them only after the field is clicked."
        )
    if flatten:
        undrawn = _flatten(ctx)
        others = 0
        for identity in undrawn:
            field_id = ctx.widget_field.get(identity)
            if field_id is not None and field_id not in failed:
                failed[field_id] = "The value could not be drawn into the flattened PDF."
            elif field_id is None:
                others += 1
        if others:
            warnings.append(f"{others} damaged form field(s) could not be drawn and were left out of the flattened PDF.")
    else:
        writer.set_need_appearances_writer(True)

    if info.encrypted:
        # Keep the author's restrictions. The owner password is random and
        # discarded: the user never had it for the original either.
        kwargs: dict[str, Any] = {}
        if info.permissions is not None:
            kwargs["permissions_flag"] = UserAccessPermissions(info.permissions)
        writer.encrypt("", secrets.token_urlsafe(32), algorithm="AES-256", **kwargs)

    buf = io.BytesIO()
    writer.write(buf)
    return FilledPdf(pdf=buf.getvalue(), failed=failed, warnings=warnings)


def _strip_incompatible(root: DictionaryObject, acro: DictionaryObject, warnings: list[str]) -> None:
    """Removes data that would contradict the new values in some viewers."""
    if "/XFA" in acro:
        del acro["/XFA"]  # hybrid form; the inspector already warned
    if "/NeedsRendering" in root:
        del root["/NeedsRendering"]
    perms = _get(root, "/Perms")
    if isinstance(perms, DictionaryObject) and ("/UR" in perms or "/UR3" in perms):
        for key in ("/UR", "/UR3"):
            perms.pop(key, None)
        if not perms:
            del root["/Perms"]
        # A rewritten file breaks the usage-rights signature, and Acrobat Reader
        # then shows an error and disables features.
        warnings.append("Removed Adobe Reader extended rights, which the filled copy can't keep.")


# --------------------------------------------------------------------------- #
# Writing values
# --------------------------------------------------------------------------- #


@dataclass
class _Located:
    node: DictionaryObject
    widgets: list[tuple[int, DictionaryObject]]  # (identity, widget)
    merged: bool  # the field dictionary is also its only widget


class _Context:
    def __init__(self, writer: PdfWriter, acro: DictionaryObject):
        self.writer = writer
        self.acro = acro
        self.fallbacks = 0
        self.page_of: dict[int, Any] = {}  # widget identity -> writer page
        self.page_index: dict[int, int] = {}  # id(page) -> 0-based index
        self.widget_field: dict[int, str] = {}  # widget identity -> id of the field written into it
        for index, page in enumerate(writer.pages):
            self.page_index[id(page)] = index
            annots = _get(page, "/Annots")
            for ref in list(annots) if isinstance(annots, ArrayObject) else []:
                self.page_of.setdefault(_identity(ref), page)
        self._shared_appearances: set[int] | None = None

    def locate(self, path: tuple[int, ...] | None) -> _Located | None:
        if not path:
            return None
        array: Any = _get(self.acro, "/Fields")
        ref: Any = None
        for index in path:
            if not isinstance(array, ArrayObject) or index >= len(array):
                return None
            ref = array[index]
            array = _get(ref.get_object(), "/Kids")
        node = ref.get_object()
        if not isinstance(node, DictionaryObject):
            return None
        kids = _get(node, "/Kids")
        if isinstance(kids, ArrayObject) and len(kids):
            widgets = [(_identity(k), k.get_object()) for k in kids if isinstance(k.get_object(), DictionaryObject)]
            return _Located(node, widgets, merged=False)
        return _Located(node, [(_identity(ref), node)], merged=True)

    def shared_appearances(self) -> set[int]:
        """Object numbers of normal appearance streams used by several widgets."""
        if self._shared_appearances is None:
            counts: dict[int, int] = {}
            for page in self.writer.pages:
                annots = _get(page, "/Annots")
                for ref in list(annots) if isinstance(annots, ArrayObject) else []:
                    ap = _get(ref.get_object(), "/AP")
                    n = ap.get("/N") if isinstance(ap, DictionaryObject) else None
                    if isinstance(n, IndirectObject):
                        counts[n.idnum] = counts.get(n.idnum, 0) + 1
            self._shared_appearances = {idnum for idnum, count in counts.items() if count > 1}
        return self._shared_appearances


def _identity(ref: Any) -> int:
    return ref.idnum if isinstance(ref, IndirectObject) else id(ref.get_object())


def _write(ctx: _Context, fld: FormField, loc: _Located, value: str, *, allow_fallback: bool) -> None:
    pages = []
    for key, _ in loc.widgets:
        page = ctx.page_of.get(key)
        if page is not None and all(page is not p for p in pages):
            pages.append(page)
    if not pages:
        raise ValueError("field has no widget on any page")

    is_button = fld.type in _BUTTONS
    node, widgets = loc.node, [w for _, w in loc.widgets]
    snapshot = _snapshot(node, widgets)
    token = f"pdfwizard_{secrets.token_hex(8)}"
    try:
        node[NameObject("/TM")] = TextStringObject(token)
        if loc.merged and "/FT" not in node:
            # pypdf only treats a widget as the field if it has its own /FT;
            # otherwise it would write to the parent (shared by siblings).
            node[NameObject("/FT")] = NameObject(_field_type_name(fld))
        if not is_button:
            _unshare_appearances(ctx, widgets)
        # pypdf draws whatever /V holds. For a combo box, draw the option's
        # display text, then store the export value.
        display = _combo_display_text(fld, node, value)
        pdf_value = "/" + value if is_button else (display or value)
        for page in pages:
            ctx.writer.update_page_form_field_values(page, {token: pdf_value}, auto_regenerate=None)
        if is_button:
            _fix_button_value(node, widgets, value)
        elif display:
            node[NameObject("/V")] = TextStringObject(value)
        elif fld.type is FieldType.CHOICE and not _inherited_int(node, "/Ff") & FF_COMBO:
            _redraw_list_box(fld, node, widgets, value)
    except Exception:
        _restore(snapshot)
        if is_button or not allow_fallback:
            raise
        # Write the value without an appearance; NeedAppearances makes viewers draw it.
        node[NameObject("/V")] = TextStringObject(value)
        node.pop("/I", None)
        ctx.fallbacks += 1
    finally:
        _restore_keys(snapshot, node, ("/TM", "/FT"))


def _field_type_name(fld: FormField) -> str:
    return {FieldType.TEXT: "/Tx", FieldType.CHOICE: "/Ch"}.get(fld.type, "/Btn")


def _combo_display_text(fld: FormField, node: DictionaryObject, value: str) -> str | None:
    if fld.type is not FieldType.CHOICE or not _inherited_int(node, "/Ff") & FF_COMBO:
        return None
    label = next((o.label for o in fld.options if o.value == value), None)
    return label if label and label != value else None


# List box drawing. Row height and highlight follow common viewer defaults.
_ROW_FACTOR = 1.15
_DEFAULT_LIST_FONT_SIZE = 12.0
_HIGHLIGHT = "0.6 0.757 0.855 rg"
# The /DA font name and fill color are copied into the new appearance stream, so
# they must stay single tokens there: ASCII only, names of printable characters
# other than delimiters, and only the whitespace PDF parsers split on. NUL is not
# accepted: parsers treat it as whitespace, so it would end a name early and let
# the rest through as operators.
_DA_WS = r"[\t\n\f\r ]+"
_DA_NAME = r"(?:(?![%()/<>\[\]{}])[!-~])+"
_DA_FONT = re.compile(rf"/({_DA_NAME}){_DA_WS}(-?\d+(?:\.\d+)?|-?\.\d+){_DA_WS}Tf", re.ASCII)
_NUM = r"(?:\d+(?:\.\d*)?|\.\d+)"
_DA_FILL = re.compile(
    rf"(?<![\w.])(?:{_NUM}{_DA_WS}){{3}}{_NUM}{_DA_WS}k\b|(?<![\w.])(?:{_NUM}{_DA_WS}){{2}}{_NUM}{_DA_WS}rg\b"
    rf"|(?<![\w.]){_NUM}{_DA_WS}g\b",
    re.ASCII,
)


def _redraw_list_box(fld: FormField, node: DictionaryObject, widgets: list[DictionaryObject], value: str) -> None:
    """Replaces pypdf's list box appearance, which puts a path inside a text
    object (invalid), never fills the selection, and doesn't scroll, so the
    selected row can be cut off. Redraws the rows, scrolled so the selection is
    visible, with the selection highlighted, and sets the top index (/TI).
    Keeps pypdf's drawing when the font isn't a simple one or a label isn't
    plain ASCII, since the text would need the font's own encoding.
    """
    labels = [o.label or o.value for o in fld.options]
    index = next((i for i, o in enumerate(fld.options) if o.value == value), None)
    if index is None or not all(label.isascii() and label.isprintable() for label in labels):
        return
    top: int | None = None
    for widget in widgets:
        stream = _normal_stream(widget)
        da = _inherited_text(widget, "/DA") or _inherited_text(node, "/DA")
        match = _DA_FONT.search(da or "")
        if stream is None or match is None or not _is_simple_font(stream, match.group(1), labels):
            continue
        x1, y1, x2, y2 = (float(v) for v in _get(stream, "/BBox"))
        width, height = x2 - x1, y2 - y1
        border = _border_width(widget)
        pad = border + 1
        if not all(math.isfinite(n) for n in (x1, y1, width, height)) or width <= 2 * pad or height <= 2 * pad:
            continue
        size = abs(float(match.group(2))) or _DEFAULT_LIST_FONT_SIZE
        size = min(size, max((height - 2 * pad) / _ROW_FACTOR, 1.0))
        row = size * _ROW_FACTOR
        rows = max(1, int((height - 2 * pad) // row))
        widget_top = 0 if index < rows else index - rows + 1
        top = widget_top if top is None else top
        fill = _DA_FILL.search(da or "")
        color = fill.group(0) if fill else "0 g"  # only a color operator from the untrusted /DA
        ops = [f"q 1 0 0 1 {x1:.2f} {y1:.2f} cm"]
        ops += _box_ops(widget, width, height, border)
        ops += ["/Tx BMC", "q", f"{pad:.2f} {pad:.2f} {width - 2 * pad:.2f} {height - 2 * pad:.2f} re W n"]
        for slot, option in enumerate(range(widget_top, min(len(labels), widget_top + rows))):
            bottom = height - pad - (slot + 1) * row
            if option == index:
                ops.append(f"q {_HIGHLIGHT} {pad:.2f} {bottom:.2f} {width - 2 * pad:.2f} {row:.2f} re f Q")
            baseline = bottom + (row - size) / 2 + 0.22 * size
            ops.append(
                f"BT /{match.group(1)} {size:.2f} Tf {color} {pad + 1:.2f} {baseline:.2f} Td ({_pdf_escape(labels[option])}) Tj ET"
            )
        ops += ["Q", "EMC", "Q"]
        stream.set_data(("\n".join(ops) + "\n").encode("latin-1"))
    if top is not None:
        node[NameObject("/TI")] = NumberObject(top)


def _normal_stream(widget: DictionaryObject) -> StreamObject | None:
    ap = _get(widget, "/AP")
    raw = ap.get("/N") if isinstance(ap, DictionaryObject) else None
    stream = raw.get_object() if raw is not None else None
    if not isinstance(stream, StreamObject) or not isinstance(_get(stream, "/BBox"), ArrayObject):
        return None
    return stream if len(_get(stream, "/BBox")) == 4 else None


def _inherited_text(node: DictionaryObject, key: str) -> str | None:
    current: Any = node
    for _ in range(32):
        if not isinstance(current, DictionaryObject):
            break
        value = _get(current, key)
        if isinstance(value, str):
            return str(value)
        current = _get(current, "/Parent")
    return None


_ASCII_ENCODINGS = ("/WinAnsiEncoding", "/StandardEncoding", "/MacRomanEncoding")


def _is_simple_font(stream: StreamObject, name: str, labels: list[str]) -> bool:
    """A Type1/TrueType font that draws these (ASCII) labels from their ASCII codes."""
    resources = _get(stream, "/Resources")
    fonts = _get(resources, "/Font") if isinstance(resources, DictionaryObject) else None
    font = _get(fonts, "/" + name) if isinstance(fonts, DictionaryObject) else None
    if not isinstance(font, DictionaryObject) or font.get("/Subtype") not in ("/Type1", "/TrueType"):
        return False
    encoding = _get(font, "/Encoding")
    if encoding is None or encoding in _ASCII_ENCODINGS:
        return True
    if not isinstance(encoding, DictionaryObject) or _get(encoding, "/BaseEncoding") not in (None, *_ASCII_ENCODINGS):
        return False
    remapped: set[int] = set()
    code = 0
    differences = _get(encoding, "/Differences")
    for item in differences if isinstance(differences, ArrayObject) else []:
        item = item.get_object()
        if isinstance(item, int):
            code = int(item)
        else:
            remapped.add(code)
            code += 1
    used = {ord(ch) for label in labels for ch in label}
    return not (used & remapped) - {39, 96}  # ' and ` are commonly remapped to their ASCII-looking glyphs


def _border_width(widget: DictionaryObject) -> float:
    bs = _get(widget, "/BS")
    try:
        width = float(_get(bs, "/W", 1)) if isinstance(bs, DictionaryObject) else 1.0
    except (TypeError, ValueError):
        width = 1.0
    return width if math.isfinite(width) and 0 <= width <= 12 else 1.0


def _box_ops(widget: DictionaryObject, width: float, height: float, border: float) -> list[str]:
    """Background (/MK /BG) and border (/MK /BC), as viewers draw them."""
    mk = _get(widget, "/MK")
    ops: list[str] = []
    if not isinstance(mk, DictionaryObject):
        return ops
    background = _color_op(_get(mk, "/BG"), stroke=False)
    if background:
        ops.append(f"q {background} 0 0 {width:.2f} {height:.2f} re f Q")
    edge = _color_op(_get(mk, "/BC"), stroke=True)
    if edge and border > 0:
        half = border / 2
        ops.append(f"q {edge} {border:.2f} w {half:.2f} {half:.2f} {width - border:.2f} {height - border:.2f} re S Q")
    return ops


def _color_op(color: Any, *, stroke: bool) -> str | None:
    if not isinstance(color, ArrayObject):
        return None
    try:
        parts = [float(v) for v in color]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(p) and 0 <= p <= 1 for p in parts):
        return None
    operator = {1: "g", 3: "rg", 4: "k"}.get(len(parts))
    if operator is None:
        return None
    return " ".join(f"{p:.3f}" for p in parts) + " " + (operator.upper() if stroke else operator)


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _inherited_int(node: DictionaryObject, key: str) -> int:
    current: Any = node
    for _ in range(32):
        if not isinstance(current, DictionaryObject):
            break
        if key in current:
            try:
                return int(_get(current, key) or 0)
            except (TypeError, ValueError):
                return 0
        current = _get(current, "/Parent")
    return 0


def _unshare_appearances(ctx: _Context, widgets: list[DictionaryObject]) -> None:
    """pypdf overwrites the existing /AP /N object in place. If that object is
    shared with other widgets (or is a direct object), drop /AP so pypdf
    creates a new appearance just for this widget."""
    shared = ctx.shared_appearances()
    for widget in widgets:
        ap = _get(widget, "/AP")
        if not isinstance(ap, DictionaryObject):
            continue
        n = ap.get("/N")
        if not isinstance(n, IndirectObject) or n.idnum in shared or not isinstance(n.get_object(), StreamObject):
            del widget["/AP"]


def _fix_button_value(node: DictionaryObject, widgets: list[DictionaryObject], state: str) -> None:
    """pypdf stores a radio group's /V as a text string and copies /V onto
    widget annotations; the spec wants a name on the field only."""
    node[NameObject("/V")] = NameObject("/" + state)
    for widget in widgets:
        if widget is not node:
            widget.pop("/V", None)


_NODE_KEYS = ("/V", "/TM", "/FT", "/I", "/TI")
_WIDGET_KEYS = ("/AS", "/V", "/AP", "/DA")


def _snapshot(node: DictionaryObject, widgets: list[DictionaryObject]) -> list[tuple[DictionaryObject, str, Any]]:
    saved = [(node, key, node.get(key, _MISSING)) for key in _NODE_KEYS]
    for widget in widgets:
        saved.extend((widget, key, widget.get(key, _MISSING)) for key in _WIDGET_KEYS)
    return saved


def _restore(snapshot) -> None:
    for obj, key, value in snapshot:
        _set_raw(obj, key, value)


def _restore_keys(snapshot, node: DictionaryObject, keys: tuple[str, ...]) -> None:
    for obj, key, value in snapshot:
        if obj is node and key in keys:
            _set_raw(obj, key, value)


def _set_raw(obj: DictionaryObject, key: str, value: Any) -> None:
    if value is _MISSING:
        obj.pop(key, None)
    else:
        obj[NameObject(key)] = value


# --------------------------------------------------------------------------- #
# Flattening
# --------------------------------------------------------------------------- #


def _flatten(ctx: _Context) -> set[int]:
    """Draws every visible widget's appearance into the page content, then
    removes the widgets and the form.

    Returns the identities of widgets that couldn't be drawn because their
    dictionaries are damaged. They are left out instead of failing the fill.
    """
    writer, acro = ctx.writer, ctx.acro
    resources_fallback = _get(acro, "/DR")
    undrawn: set[int] = set()
    for page_number, page in enumerate(writer.pages):
        annots = _get(page, "/Annots")
        if not isinstance(annots, ArrayObject):
            continue
        ops: list[str] = []
        keep = ArrayObject()
        xobjects: DictionaryObject | None = None
        for ref in list(annots):
            try:
                annot = ref.get_object()
                is_widget = isinstance(annot, DictionaryObject) and annot.get("/Subtype") == "/Widget"
            except Exception as exc:  # unreadable entry: not ours to drop
                log.info("could not read an annotation: %s", type(exc).__name__)
                keep.append(ref)
                continue
            if not is_widget:
                keep.append(ref)
                continue
            try:
                drawn = _appearance(annot)
                if drawn is None:
                    continue
                stream_ref, stream = drawn
                name = f"/PdfWizardFm{page_number}_{len(ops)}"
                op = _placement(annot, stream, name)
                if xobjects is None:
                    xobjects = _page_xobjects(page)
                if "/Resources" not in stream and isinstance(resources_fallback, DictionaryObject):
                    stream[NameObject("/Resources")] = resources_fallback
                stream[NameObject("/Subtype")] = NameObject("/Form")
                xobjects[NameObject(name)] = stream_ref if isinstance(stream_ref, IndirectObject) else writer._add_object(stream)
                ops.append(op)
            except Exception as exc:  # malformed /F, /AP, /BBox, /Matrix or /Rect (pypdf raises many types)
                log.info("could not flatten a widget: %s", type(exc).__name__)
                undrawn.add(_identity(ref))
        if ops:
            _append_content(writer, page, "".join(ops))
        if keep:
            page[NameObject("/Annots")] = keep
        else:
            del page["/Annots"]
    del writer.root_object["/AcroForm"]
    return undrawn


def _appearance(annot: DictionaryObject) -> tuple[Any, StreamObject] | None:
    flags = int(_get(annot, "/F", 0) or 0)
    if flags & (ANNOT_HIDDEN | ANNOT_NO_VIEW):
        return None
    ap = _get(annot, "/AP")
    if not isinstance(ap, DictionaryObject):
        return None
    raw = ap.get("/N")
    normal = raw.get_object() if raw is not None else None
    if isinstance(normal, DictionaryObject) and not isinstance(normal, StreamObject):
        state = annot.get("/AS")
        raw = normal.get(state) if state is not None else None
        normal = raw.get_object() if raw is not None else None
    if not isinstance(normal, StreamObject) or "/BBox" not in normal:
        return None
    return raw, normal


def _placement(annot: DictionaryObject, stream: StreamObject, name: str) -> str:
    """Maps the appearance's transformed /BBox onto the widget /Rect (ISO 32000-1, 12.5.5).

    Raises ValueError or TypeError when /BBox, /Matrix or /Rect is malformed.
    """
    bx1, by1, bx2, by2 = (float(v) for v in _get(stream, "/BBox"))
    matrix = _get(stream, "/Matrix")
    a, b, c, d, e, f = (float(v) for v in matrix) if isinstance(matrix, ArrayObject) and len(matrix) == 6 else (1, 0, 0, 1, 0, 0)
    xs, ys = [], []
    for x, y in ((bx1, by1), (bx2, by1), (bx1, by2), (bx2, by2)):
        xs.append(a * x + c * y + e)
        ys.append(b * x + d * y + f)
    rx1, ry1, rx2, ry2 = (float(v) for v in _get(annot, "/Rect"))
    rx1, rx2 = sorted((rx1, rx2))
    ry1, ry2 = sorted((ry1, ry2))
    width, height = max(xs) - min(xs), max(ys) - min(ys)
    sx = (rx2 - rx1) / width if width > 0 else 1.0
    sy = (ry2 - ry1) / height if height > 0 else 1.0
    tx, ty = rx1 - min(xs) * sx, ry1 - min(ys) * sy
    numbers = [sx, 0.0, 0.0, sy, tx, ty]
    if not all(math.isfinite(n) for n in numbers):
        raise ValueError("non-finite placement")
    return "q " + " ".join(f"{n:.5f}" for n in numbers) + f" cm {name} Do Q\n"


def _page_xobjects(page: DictionaryObject) -> DictionaryObject:
    """The page's own /Resources /XObject dict, created without losing inherited entries."""
    resources = _get(page, "/Resources")
    if resources is None:
        inherited = page.get_inherited("/Resources", None)
        resources = DictionaryObject(inherited.get_object()) if inherited is not None else DictionaryObject()
        page[NameObject("/Resources")] = resources
    xobjects = _get(resources, "/XObject")
    if not isinstance(xobjects, DictionaryObject):
        xobjects = DictionaryObject()
    else:
        xobjects = DictionaryObject(xobjects)  # don't modify a dict shared with other pages
    resources[NameObject("/XObject")] = xobjects
    return xobjects


def _append_content(writer: PdfWriter, page: DictionaryObject, ops: str) -> None:
    """Wraps the existing content in q/Q (it may leave the graphics state
    changed), then draws the appearances."""
    existing = page.get("/Contents")
    resolved = existing.get_object() if existing is not None else None
    if isinstance(resolved, ArrayObject):
        streams = list(resolved)
    elif existing is not None:
        streams = [existing]
    else:
        streams = []
    page[NameObject("/Contents")] = ArrayObject(
        [_stream(writer, b"q\n"), *streams, _stream(writer, b"\nQ\n" + ops.encode("latin-1"))]
    )


def _stream(writer: PdfWriter, data: bytes) -> IndirectObject:
    stream = DecodedStreamObject()
    stream.set_data(data)
    return writer._add_object(stream)
