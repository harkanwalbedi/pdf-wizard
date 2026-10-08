"""Builds small PDFs for tests. reportlab (and its dependency Pillow) are dev-only."""

from __future__ import annotations

import io
import zlib
from dataclasses import dataclass, field
from typing import Literal

from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject,
    DecodedStreamObject,
    DictionaryObject,
    EncodedStreamObject,
    FloatObject,
    NameObject,
    NumberObject,
    TextStringObject,
)
from reportlab import rl_config
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

rl_config.invariant = 1  # reproducible bytes: no timestamps or random document IDs


@dataclass
class Spec:
    """One field of a generated form.

    For radio groups, add one Spec per button with the same `name` and a
    different `value` (the export value).
    """

    kind: Literal["text", "checkbox", "radio", "choice", "listbox"]
    name: str
    label: str | None = None  # printed left of the field (right of checkboxes/radios)
    tooltip: str | None = None  # becomes /TU
    x: float = 200
    y: float = 700
    width: float = 220
    height: float = 18
    value: str = ""  # text/choice initial value; radio: this button's export value
    max_len: int | None = None
    flags: str = ""  # reportlab fieldFlags: "readOnly", "multiline", "comb", "combo edit", ...
    options: list = field(default_factory=list)  # choice: ["A", ...] or [("Label", "export"), ...]
    checked: bool = False  # checkbox initial state / selected radio button
    page: int = 1


def make_form(specs: list[Spec], *, pages: int = 1, header: str | None = None) -> bytes:
    """An AcroForm PDF with the given fields and their printed labels."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    for page in range(1, pages + 1):
        c.setFont("Helvetica", 10)
        if header:
            c.drawString(72, 750, header)
        for spec in specs:
            if spec.page == page:
                _draw_field(c, spec)
        c.showPage()
    c.save()
    return buf.getvalue()


def _draw_field(c: canvas.Canvas, s: Spec) -> None:
    form = c.acroForm
    common = {"name": s.name, "tooltip": s.tooltip, "x": s.x, "y": s.y, "forceBorder": True}
    if s.kind == "text":
        if s.label:
            c.drawRightString(s.x - 6, s.y + 5, s.label)
        form.textfield(
            value=s.value, width=s.width, height=s.height, maxlen=s.max_len, fieldFlags=s.flags, **common
        )
    elif s.kind == "checkbox":
        if s.label:
            c.drawString(s.x + s.height + 6, s.y + 4, s.label)
        form.checkbox(checked=s.checked, size=s.height, fieldFlags=s.flags, **common)
    elif s.kind == "radio":
        if s.label:
            c.drawString(s.x + s.height + 6, s.y + 4, s.label)
        form.radio(value=s.value, selected=s.checked, size=s.height, **common)
    else:
        if s.label:
            c.drawRightString(s.x - 6, s.y + 5, s.label)
        method = form.choice if s.kind == "choice" else form.listbox
        flags = s.flags or ("combo" if s.kind == "choice" else "")
        method(value=s.value, options=s.options, width=s.width, height=s.height, fieldFlags=flags, **common)


def make_text_pdf(lines: list[str], *, pages: int = 1) -> bytes:
    """A plain PDF document (no form) whose text layer contains `lines` on every page."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    for _ in range(pages):
        c.setFont("Helvetica", 11)
        y = 740
        for line in lines:
            c.drawString(72, y, line)
            y -= 16
        c.showPage()
    c.save()
    return buf.getvalue()


def make_blank_pdf(pages: int = 1) -> bytes:
    """A PDF with empty pages and no form."""
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(612, 792)
    return to_bytes(writer)


def make_png(width: int = 8, height: int = 8) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


def make_jpeg(width: int = 8, height: int = 8) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), (30, 30, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def to_bytes(writer: PdfWriter) -> bytes:
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _edit(pdf: bytes) -> PdfWriter:
    return PdfWriter(clone_from=PdfReader(io.BytesIO(pdf)))


def _widgets(writer: PdfWriter):
    for page in writer.pages:
        for ref in page.get("/Annots", []) or []:
            annot = ref.get_object()
            if annot.get("/Subtype") == "/Widget":
                yield annot


def set_scripts(pdf: bytes, field_name: str, *, format_js: str | None = None, keystroke_js: str | None = None) -> bytes:
    """Adds Acrobat format (/AA /F) and keystroke (/AA /K) JavaScript to a field."""
    writer = _edit(pdf)
    for annot in _widgets(writer):
        target = annot if "/T" in annot else annot["/Parent"].get_object()
        if target.get("/T") != field_name:
            continue
        aa = DictionaryObject()
        if format_js is not None:
            aa[NameObject("/F")] = _js_action(format_js)
        if keystroke_js is not None:
            aa[NameObject("/K")] = _js_action(keystroke_js)
        target[NameObject("/AA")] = aa
    return to_bytes(writer)


def _js_action(js: str) -> DictionaryObject:
    return DictionaryObject(
        {NameObject("/S"): NameObject("/JavaScript"), NameObject("/JS"): TextStringObject(js)}
    )


def set_field_keys(pdf: bytes, field_name: str, **keys) -> bytes:
    """Sets raw keys on a terminal field, e.g. set_field_keys(pdf, "a", Ff=1)."""
    writer = _edit(pdf)
    for annot in _widgets(writer):
        target = annot if "/T" in annot else annot["/Parent"].get_object()
        if target.get("/T") == field_name:
            for key, value in keys.items():
                target[NameObject("/" + key)] = _pdf_value(value)
    return to_bytes(writer)


def set_widget_flags(pdf: bytes, field_name: str, flags: int) -> bytes:
    """Sets the annotation flags (/F) of every widget of a field (2 = hidden)."""
    writer = _edit(pdf)
    for annot in _widgets(writer):
        target = annot if "/T" in annot else annot["/Parent"].get_object()
        if target.get("/T") == field_name:
            annot[NameObject("/F")] = NumberObject(flags)
    return to_bytes(writer)


def _pdf_value(value):
    if isinstance(value, bool):
        raise TypeError("use ints for flags")
    if isinstance(value, int):
        return NumberObject(value)
    if isinstance(value, float):
        return FloatObject(value)
    if isinstance(value, str) and value.startswith("/"):
        return NameObject(value)
    return TextStringObject(value)


def encrypt_pdf(pdf: bytes, *, user_password: str = "", owner_password: str = "owner-secret",
                permissions: int = -1) -> bytes:
    writer = _edit(pdf)
    writer.encrypt(user_password, owner_password, permissions_flag=permissions, algorithm="AES-256")
    return to_bytes(writer)


def add_xfa(pdf: bytes, *, remove_fields: bool = False) -> bytes:
    """Adds an /XFA entry. With remove_fields, makes a pure XFA form (no AcroForm fields)."""
    writer = _edit(pdf)
    acro = writer.root_object["/AcroForm"]
    stream = DecodedStreamObject()
    stream.set_data(b"<xdp:xdp xmlns:xdp='http://ns.adobe.com/xdp/'></xdp:xdp>")
    acro[NameObject("/XFA")] = writer._add_object(stream)
    if remove_fields:
        acro[NameObject("/Fields")] = ArrayObject()
        writer.remove_annotations(["/Widget"])
    return to_bytes(writer)


def make_hierarchical_form() -> bytes:
    """A hand-built form with field hierarchies that trip up name-based filling.

    - "Name": top-level text field (field and widget merged).
    - "Spouse.Name": the kid is a merged field/widget whose /FT is inherited
      from its parent, so the kid itself has no /FT.
    - "Address.City": non-terminal parent with a merged text kid.
    - "Phone": one text field with two widgets (no /T on the widgets).
    - "Agree": a checkbox nested under "Terms".
    """
    writer = PdfWriter()
    page = writer.add_blank_page(612, 792)
    helv = writer._add_object(
        DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
                NameObject("/Encoding"): NameObject("/WinAnsiEncoding"),
            }
        )
    )
    fields = ArrayObject()
    writer.root_object[NameObject("/AcroForm")] = DictionaryObject(
        {
            NameObject("/Fields"): fields,
            NameObject("/DA"): TextStringObject("/Helv 0 Tf 0 g"),
            NameObject("/DR"): DictionaryObject(
                {NameObject("/Font"): DictionaryObject({NameObject("/Helv"): helv})}
            ),
        }
    )
    annots = ArrayObject()
    page[NameObject("/Annots")] = annots

    def widget(rect, **extra):
        d = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Annot"),
                NameObject("/Subtype"): NameObject("/Widget"),
                NameObject("/Rect"): ArrayObject([FloatObject(v) for v in rect]),
                NameObject("/F"): NumberObject(4),
                NameObject("/P"): page.indirect_reference,
                NameObject("/DA"): TextStringObject("/Helv 10 Tf 0 g"),
            }
        )
        for key, value in extra.items():
            d[NameObject("/" + key)] = _pdf_value(value)
        return d

    def node(**keys):
        d = DictionaryObject()
        for key, value in keys.items():
            d[NameObject("/" + key)] = _pdf_value(value)
        return d

    # Name
    name = writer._add_object(widget([200, 700, 420, 718], FT="/Tx", T="Name", TU="Your name"))
    fields.append(name)
    annots.append(name)

    # Spouse.Name (kid inherits /FT)
    spouse = writer._add_object(node(T="Spouse", FT="/Tx"))
    spouse_name = writer._add_object(widget([200, 660, 420, 678], T="Name", TU="Spouse name"))
    spouse_name.get_object()[NameObject("/Parent")] = spouse
    spouse.get_object()[NameObject("/Kids")] = ArrayObject([spouse_name])
    fields.append(spouse)
    annots.append(spouse_name)

    # Address.City
    address = writer._add_object(node(T="Address"))
    city = writer._add_object(widget([200, 620, 420, 638], FT="/Tx", T="City", TU="City"))
    city.get_object()[NameObject("/Parent")] = address
    address.get_object()[NameObject("/Kids")] = ArrayObject([city])
    fields.append(address)
    annots.append(city)

    # Phone: one field, two widgets
    phone = writer._add_object(node(T="Phone", FT="/Tx", TU="Phone number"))
    kids = ArrayObject()
    for rect in ([200, 580, 420, 598], [200, 540, 420, 558]):
        w = writer._add_object(widget(rect))
        w.get_object()[NameObject("/Parent")] = phone
        kids.append(w)
        annots.append(w)
    phone.get_object()[NameObject("/Kids")] = kids
    fields.append(phone)

    # Terms.Agree checkbox
    terms = writer._add_object(node(T="Terms"))
    agree = widget([200, 500, 214, 514], FT="/Btn", T="Agree", TU="I agree", V="/Off", AS="/Off")
    on, off = DecodedStreamObject(), DecodedStreamObject()
    on.set_data(b"0 0 1 rg 2 2 10 10 re f")
    off.set_data(b"")
    for s in (on, off):
        s[NameObject("/Type")] = NameObject("/XObject")
        s[NameObject("/Subtype")] = NameObject("/Form")
        s[NameObject("/BBox")] = ArrayObject([FloatObject(0), FloatObject(0), FloatObject(14), FloatObject(14)])
    agree[NameObject("/AP")] = DictionaryObject(
        {
            NameObject("/N"): DictionaryObject(
                {NameObject("/Yes"): writer._add_object(on), NameObject("/Off"): writer._add_object(off)}
            )
        }
    )
    agree_ref = writer._add_object(agree)
    agree_ref.get_object()[NameObject("/Parent")] = terms
    terms.get_object()[NameObject("/Kids")] = ArrayObject([agree_ref])
    fields.append(terms)
    annots.append(agree_ref)

    return to_bytes(writer)


def sample_form() -> bytes:
    """A one-page form covering every supported field type, used across tests."""
    return make_form(
        [
            Spec("text", "full_name", label="Full name", y=700),
            Spec("text", "dob", label="Date of birth", y=670, width=100),
            Spec("text", "ssn", label="SSN", y=640, width=120, max_len=9, flags="comb"),
            Spec("text", "email", tooltip="Email address", y=610),
            Spec("text", "notes", label="Notes", y=540, height=50, flags="multiline", max_len=None),
            Spec("text", "office_use", label="Office use only", y=510, flags="readOnly"),
            Spec("checkbox", "married", label="Married", x=200, y=480, height=12),
            Spec("radio", "employment", label="Employed", value="employed", x=200, y=450, height=12),
            Spec("radio", "employment", label="Self-employed", value="self", x=320, y=450, height=12),
            Spec("choice", "state", label="State", y=420, width=120,
                 options=[("California", "CA"), ("Texas", "TX"), ("New York", "NY")], value="CA"),
            Spec("listbox", "color", label="Favorite color", y=360, width=120, height=40,
                 options=["Red", "Green", "Blue"], value="Red"),
        ]
    )


# --------------------------------------------------------------------------- #
# Hostile inputs for the parser limits
# --------------------------------------------------------------------------- #


def flate_stream(data: bytes) -> EncodedStreamObject:
    """A FlateDecode stream holding `data`; pypdf inflates it only when asked."""
    raw = DecodedStreamObject()
    raw.set_data(data)
    return raw.flate_encode(level=9)


def set_field_stream(pdf: bytes, field_name: str, key: str, data: bytes) -> bytes:
    """Sets a terminal field's `key` ("V", "TU", ...) to a FlateDecode stream of `data`."""
    writer = _edit(pdf)
    stream = writer._add_object(flate_stream(data))
    for annot in _widgets(writer):
        target = annot if "/T" in annot else annot["/Parent"].get_object()
        if target.get("/T") == field_name:
            target[NameObject("/" + key)] = stream
    return to_bytes(writer)


def set_script_stream(pdf: bytes, field_name: str, script: bytes, *, compress: bool = True) -> bytes:
    """Adds a format script (/AA /F) whose /JS is a stream, Flate-encoded unless compress=False."""
    writer = _edit(pdf)
    if compress:
        stream = flate_stream(script)
    else:
        stream = DecodedStreamObject()
        stream.set_data(script)
    action = DictionaryObject(
        {NameObject("/S"): NameObject("/JavaScript"), NameObject("/JS"): writer._add_object(stream)}
    )
    for annot in _widgets(writer):
        target = annot if "/T" in annot else annot["/Parent"].get_object()
        if target.get("/T") == field_name:
            target[NameObject("/AA")] = DictionaryObject({NameObject("/F"): action})
    return to_bytes(writer)


def make_state_heavy_form(fields: int, states: int, runs: int, *, heading_size: float | None = None) -> bytes:
    """Checkboxes sharing one /AP /N dictionary with `states` on-states, on a page
    with `runs` one-letter text runs spaced so they never merge into lines.

    `heading_size` adds one line in that font size at the top of the page, which
    widens the band of lines every label search has to look at.
    """
    writer = PdfWriter()
    page = writer.add_blank_page(612, 792)
    font = writer._add_object(
        DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    ops = [f"BT /F1 {heading_size} Tf 1 0 0 1 10 790 Tm (H) Tj ET"] if heading_size else []
    ops += [f"BT /F1 1 Tf 1 0 0 1 {10 + (i % 190) * 3} {780 - (i // 190) * 3} Tm (a) Tj ET" for i in range(runs)]
    content = DecodedStreamObject()
    content.set_data("\n".join(ops).encode())
    page[NameObject("/Contents")] = writer._add_object(content.flate_encode())

    blank = DecodedStreamObject()
    blank.set_data(b"")
    blank[NameObject("/BBox")] = ArrayObject([FloatObject(0), FloatObject(0), FloatObject(10), FloatObject(10)])
    blank_ref = writer._add_object(blank)
    normal = DictionaryObject({NameObject(f"/S{i}"): blank_ref for i in range(states)})
    normal[NameObject("/Off")] = blank_ref
    ap = writer._add_object(DictionaryObject({NameObject("/N"): writer._add_object(normal)}))

    fields_array, annots = ArrayObject(), ArrayObject()
    for i in range(fields):
        x, y = 20 + (i % 50) * 11, 20 + (i // 50) * 11
        widget = writer._add_object(
            DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Annot"),
                    NameObject("/Subtype"): NameObject("/Widget"),
                    NameObject("/FT"): NameObject("/Btn"),
                    NameObject("/T"): TextStringObject(f"c{i}"),
                    NameObject("/Rect"): ArrayObject([FloatObject(v) for v in (x, y, x + 10, y + 10)]),
                    NameObject("/F"): NumberObject(4),
                    NameObject("/P"): page.indirect_reference,
                    NameObject("/AP"): ap,
                    NameObject("/AS"): NameObject("/Off"),
                }
            )
        )
        fields_array.append(widget)
        annots.append(widget)
    page[NameObject("/Annots")] = annots
    writer.root_object[NameObject("/AcroForm")] = DictionaryObject({NameObject("/Fields"): fields_array})
    return to_bytes(writer)


def make_object_stream_bomb(inflated: int) -> bytes:
    """A one-page PDF whose /AcroForm sits in an object stream that inflates to `inflated` bytes."""
    header, acroform = b"5 0 ", b"<</Fields[]>>"
    stream = zlib.compress(header + acroform + b" " * (inflated - len(header) - len(acroform)), 9)
    objects = [
        b"<</Type/Catalog/Pages 2 0 R/AcroForm 5 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>",
        b"<</Type/ObjStm/N 1/First %d/Filter/FlateDecode/Length %d>>stream\n%s\nendstream"
        % (len(header), len(stream), stream),
    ]
    out = bytearray(b"%PDF-1.5\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref_offset = len(out)
    # Cross-reference stream: objects 1-4 and 6 (itself) by file offset, object 5 inside stream 4.
    rows = [(0, 0, 255), *((1, offset, 0) for offset in offsets), (2, 4, 0), (1, xref_offset, 0)]
    table = b"".join(bytes([kind]) + value.to_bytes(4, "big") + bytes([extra]) for kind, value, extra in rows)
    out += b"6 0 obj\n<</Type/XRef/Size 7/W[1 4 1]/Root 1 0 R/Length %d>>stream\n%s\nendstream\nendobj\n" % (
        len(table),
        table,
    )
    out += b"startxref\n%d\n%%%%EOF\n" % xref_offset
    return bytes(out)


def encrypt_with_page_count(pdf: bytes, count: int) -> bytes:
    """Encrypts `pdf` (empty user password) with /Pages /Count declaring `count` pages."""
    writer = _edit(pdf)
    writer.root_object["/Pages"][NameObject("/Count")] = NumberObject(count)
    writer.encrypt("", "owner-secret", algorithm="AES-256")
    return to_bytes(writer)
