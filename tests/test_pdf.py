"""Tests for the PDF inspector and filler."""

from __future__ import annotations

import io
import itertools
import re
from concurrent.futures import ThreadPoolExecutor

import pypdf
import pytest
from pypdf import PageObject, PdfReader, PdfWriter
from pypdf.errors import LimitReachedError
from pypdf.generic import (
    ArrayObject,
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    NumberObject,
    TextStringObject,
)

import pdf_factory as pf
from pdf_wizard.config import Limits
from pdf_wizard.errors import InvalidInputError, UnsupportedFormError
from pdf_wizard.models import FieldType
from pdf_wizard.pdf import filler, inspector
from pdf_wizard.pdf.filler import fill_pdf
from pdf_wizard.pdf.inspector import open_form, parse_format_script
from pdf_wizard.pdf.limits import MAX_CONTENT_STREAMS, MAX_STREAM_BYTES, page_text
from pdf_wizard.sources.common import Context
from pdf_wizard.sources.pdfs import load_pdf_document


def by_name(form):
    return {f.name: f for f in form.info.fields}


def reread(pdf: bytes):
    return by_name(open_form(pdf, for_filling=False))


def raw_fields(pdf: bytes, password: str | None = None):
    reader = PdfReader(io.BytesIO(pdf))
    if password is not None:
        reader.decrypt(password)
    return reader


# --------------------------------------------------------------------------- #
# Inspector
# --------------------------------------------------------------------------- #


def test_inspects_every_field_type():
    form = open_form(pf.sample_form())
    fields = by_name(form)
    assert [f.id for f in form.info.fields] == [f"F{i}" for i in range(1, 11)]
    assert fields["full_name"].type is FieldType.TEXT
    assert fields["full_name"].label == "Full name"
    assert fields["dob"].label == "Date of birth"
    assert fields["email"].label == "Email address"  # from the tooltip
    assert fields["ssn"].max_len == 9 and fields["ssn"].comb
    assert fields["notes"].multiline
    assert fields["office_use"].read_only and not fields["office_use"].writable
    married = fields["married"]
    assert married.type is FieldType.CHECKBOX
    assert [(o.value, o.label) for o in married.options] == [("Yes", "Married")]
    assert married.current_value == "Off"
    employment = fields["employment"]
    assert employment.type is FieldType.RADIO
    assert [(o.value, o.label) for o in employment.options] == [("employed", "Employed"), ("self", "Self-employed")]
    state = fields["state"]
    assert state.type is FieldType.CHOICE and state.current_value == "CA"
    assert [(o.value, o.label) for o in state.options][:2] == [("CA", "California"), ("TX", "Texas")]
    assert fields["color"].options[0].value == "Red"
    assert all(f.page == 1 and f.rect for f in form.info.fields)
    assert form.info.page_count == 1 and len(form.info.sha256) == 64
    assert not form.info.encrypted and not form.info.had_xfa


def test_hierarchical_names_and_inherited_type():
    fields = by_name(open_form(pf.make_hierarchical_form()))
    assert list(fields) == ["Name", "Spouse.Name", "Address.City", "Phone", "Terms.Agree"]
    assert fields["Spouse.Name"].type is FieldType.TEXT  # /FT inherited from the parent
    assert fields["Terms.Agree"].type is FieldType.CHECKBOX
    assert fields["Phone"].label == "Phone number"


def test_radio_group_label_is_the_text_left_of_the_first_button():
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.setFont("Helvetica", 10)
    c.drawRightString(192, 704, "Medical plan")
    for i, (label, value) in enumerate([("Basic", "basic"), ("Premium", "premium")]):
        c.acroForm.radio(name="plan", value=value, x=200 + 100 * i, y=700, size=12, forceBorder=True)
        c.drawString(216 + 100 * i, 703, label)
    # Option labels printed left of each button must not become the group label.
    c.drawString(72, 654, "Citizen?")
    for i, (label, value) in enumerate([("Yes", "Y"), ("No", "N")]):
        c.drawRightString(196 + 100 * i, 654, label)
        c.acroForm.radio(name="citizen", value=value, x=200 + 100 * i, y=650, size=12, forceBorder=True)
    c.showPage()
    c.save()

    fields = by_name(open_form(buf.getvalue()))
    assert fields["plan"].label == "Medical plan"
    assert [o.label for o in fields["plan"].options] == ["Basic", "Premium"]
    assert [o.label for o in fields["citizen"].options] == ["Yes", "No"]
    assert fields["citizen"].label == "Citizen?"
    assert by_name(open_form(pf.sample_form()))["employment"].label is None  # nothing printed left of it


@pytest.mark.parametrize(
    ("script", "hint"),
    [
        ('AFDate_FormatEx("mm/dd/yyyy");', "date:mm/dd/yyyy"),
        ("AFDate_KeystrokeEx('yyyy-mm-dd')", "date:yyyy-mm-dd"),
        ("AFDate_Format(2);", "date:mm/dd/yy"),
        ("AFSpecial_Format(0);", "zip"),
        ("AFSpecial_Keystroke(1);", "zip4"),
        ("AFSpecial_Format(2);", "phone"),
        ("AFSpecial_Format(3);", "ssn"),
        ('AFSpecial_KeystrokeEx("999-99-9999");', "mask:999-99-9999"),
        ('AFNumber_Format(2, 0, 0, 0, "$", true);', "number:2:0"),
        ("AFPercent_Format(1, 0);", "percent:1:0"),
        ("event.value = util.printd('mm', new Date());", None),
        ("", None),
    ],
)
def test_parse_format_script(script, hint):
    assert parse_format_script(script) == hint


def test_format_hint_from_field_scripts():
    pdf = pf.set_scripts(pf.sample_form(), "dob", format_js='AFDate_FormatEx("mm/dd/yyyy");')
    pdf = pf.set_scripts(pdf, "full_name", keystroke_js="AFSpecial_Keystroke(0);")
    fields = by_name(open_form(pdf))
    assert fields["dob"].format_hint == "date:mm/dd/yyyy"
    assert fields["full_name"].format_hint == "zip"  # keystroke script used when there's no format script
    assert fields["email"].format_hint is None


def test_hidden_and_password_fields_are_not_writable():
    pdf = pf.set_widget_flags(pf.sample_form(), "email", 2)
    pdf = pf.set_field_keys(pdf, "full_name", Ff=1 << 13)
    fields = by_name(open_form(pdf))
    assert fields["email"].hidden and not fields["email"].writable
    assert fields["full_name"].password and not fields["full_name"].writable


@pytest.mark.parametrize(
    "data",
    [b"", b"hello world", b"\x89PNG\r\n\x1a\n" + b"\x00" * 100, pf.sample_form()[:400]],
    ids=["empty", "text", "png", "truncated"],
)
def test_rejects_non_pdf_and_damaged_input(data):
    with pytest.raises(InvalidInputError):
        open_form(data)


def test_size_and_page_caps():
    with pytest.raises(InvalidInputError, match="too large"):
        open_form(pf.sample_form(), limits=Limits(max_file_bytes=1000))
    two_pages = pf.make_form([pf.Spec("text", "a"), pf.Spec("text", "b", page=2)], pages=2)
    with pytest.raises(InvalidInputError, match="too many pages"):
        open_form(two_pages, limits=Limits(max_pages=1))


def test_field_cap_warns():
    form = open_form(pf.sample_form(), limits=Limits(max_fields=3))
    assert len(form.info.fields) == 3
    assert any("only the first 3" in w for w in form.info.warnings)


def test_encrypted_with_empty_password_opens():
    pdf = pf.encrypt_pdf(pf.sample_form(), permissions=4 | 256)
    form = open_form(pdf)
    assert form.info.encrypted and form.info.permissions == 4 | 256
    plain = form.plain_pdf()
    assert not PdfReader(io.BytesIO(plain)).is_encrypted


def test_password_protected_is_rejected():
    pdf = pf.encrypt_pdf(pf.sample_form(), user_password="secret")
    with pytest.raises(UnsupportedFormError, match="password"):
        open_form(pdf)
    with pytest.raises(UnsupportedFormError):
        open_form(pdf, for_filling=False)


def test_permissions_that_forbid_filling():
    pdf = pf.encrypt_pdf(pf.sample_form(), permissions=4)  # print only
    with pytest.raises(UnsupportedFormError, match="security settings"):
        open_form(pdf)
    assert open_form(pdf, for_filling=False).info.fields  # still readable as a reference


def test_no_fields():
    pdf = pf.make_text_pdf(["Just text"])
    with pytest.raises(UnsupportedFormError, match="no fillable"):
        open_form(pdf)
    assert open_form(pdf, for_filling=False).info.fields == []


def test_xfa_forms():
    with pytest.raises(UnsupportedFormError, match="XFA"):
        open_form(pf.add_xfa(pf.sample_form(), remove_fields=True))
    hybrid = open_form(pf.add_xfa(pf.sample_form()))
    assert hybrid.info.had_xfa and any("XFA" in w for w in hybrid.info.warnings)


def test_cyclic_field_tree_terminates():
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(pf.make_hierarchical_form())))
    fields = writer.root_object["/AcroForm"]["/Fields"]
    address_ref = fields[2]
    address = address_ref.get_object()
    address["/Kids"].append(address_ref)  # Address is its own kid
    city = address["/Kids"][0].get_object()
    city[NameObject("/Kids")] = ArrayObject([address_ref])  # and its grandchild
    buf = io.BytesIO()
    writer.write(buf)
    form = open_form(buf.getvalue())
    assert "Name" in by_name(form)


# --------------------------------------------------------------------------- #
# Filler
# --------------------------------------------------------------------------- #


def fill_by_name(pdf: bytes, values: dict[str, str], **kwargs):
    form = open_form(pdf)
    ids = {f.name: f.id for f in form.info.fields}
    return form, fill_pdf(form, {ids[name]: value for name, value in values.items()}, **kwargs)


def test_fill_round_trip():
    _, result = fill_by_name(
        pf.sample_form(),
        {
            "full_name": "Jane Q. Public",
            "ssn": "123456789",
            "notes": "Line one\nLine two",
            "married": "Yes",
            "employment": "self",
            "state": "TX",
            "color": "Blue",
        },
    )
    assert result.failed == {} and result.warnings == []
    fields = reread(result.pdf)
    assert fields["full_name"].current_value == "Jane Q. Public"
    assert fields["ssn"].current_value == "123456789"
    assert fields["notes"].current_value == "Line one\nLine two"
    assert fields["married"].current_value == "Yes"
    assert fields["employment"].current_value == "self"
    assert fields["state"].current_value == "TX"
    assert fields["color"].current_value == "Blue"
    assert fields["email"].current_value is None  # untouched
    reader = raw_fields(result.pdf)
    assert reader.root_object["/AcroForm"]["/NeedAppearances"] is True or reader.root_object["/AcroForm"]["/NeedAppearances"]


def test_radio_and_checkbox_use_names_and_states():
    _, result = fill_by_name(pf.sample_form(), {"employment": "self", "married": "Yes"})
    reader = raw_fields(result.pdf)
    employment = next(f for f in reader.root_object["/AcroForm"]["/Fields"] if f.get_object()["/T"] == "employment")
    employment = employment.get_object()
    assert isinstance(employment["/V"], NameObject) and employment["/V"] == "/self"
    states = []
    for kid in employment["/Kids"]:
        kid = kid.get_object()
        assert "/V" not in kid
        states.append(kid["/AS"])
    assert states == ["/Off", "/self"]


def test_uncheck_a_checked_box():
    pdf = pf.make_form([pf.Spec("checkbox", "agree", checked=True)])
    assert by_name(open_form(pdf))["agree"].current_value == "Yes"
    _, result = fill_by_name(pdf, {"agree": "Off"})
    assert reread(result.pdf)["agree"].current_value == "Off"


def test_partial_name_collisions_do_not_leak():
    _, result = fill_by_name(pf.make_hierarchical_form(), {"Name": "Alice"})
    fields = reread(result.pdf)
    assert fields["Name"].current_value == "Alice"
    assert fields["Spouse.Name"].current_value is None

    _, result = fill_by_name(pf.make_hierarchical_form(), {"Spouse.Name": "Bob", "Address.City": "Paris"})
    fields = reread(result.pdf)
    assert fields["Name"].current_value is None
    assert fields["Spouse.Name"].current_value == "Bob"
    assert fields["Address.City"].current_value == "Paris"
    reader = raw_fields(result.pdf)
    for annot in reader.pages[0]["/Annots"]:
        annot = annot.get_object()
        assert "/TM" not in annot  # temporary names removed
        if annot.get("/T") == "Name" and "/Parent" in annot:
            assert "/FT" not in annot  # temporary /FT removed; still inherited


def test_field_with_two_widgets_gets_both_appearances():
    _, result = fill_by_name(pf.make_hierarchical_form(), {"Phone": "555-1234"})
    reader = raw_fields(result.pdf)
    phone = next(f.get_object() for f in reader.root_object["/AcroForm"]["/Fields"] if f.get_object()["/T"] == "Phone")
    for kid in phone["/Kids"]:
        assert b"555-1234" in kid.get_object()["/AP"]["/N"].get_object().get_data()


def test_shared_appearance_stream_is_not_overwritten():
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(pf.sample_form())))
    annots = [a.get_object() for a in writer.pages[0]["/Annots"]]
    full_name = next(a for a in annots if a.get("/T") == "full_name")
    email = next(a for a in annots if a.get("/T") == "email")
    email["/AP"][NameObject("/N")] = full_name["/AP"].raw_get("/N")  # share one stream
    buf = io.BytesIO()
    writer.write(buf)

    _, result = fill_by_name(buf.getvalue(), {"full_name": "Jane"})
    reader = raw_fields(result.pdf)
    annots = {a.get_object().get("/T"): a.get_object() for a in reader.pages[0]["/Annots"]}
    assert b"Jane" in annots["full_name"]["/AP"]["/N"].get_object().get_data()
    assert b"Jane" not in annots["email"]["/AP"]["/N"].get_object().get_data()


def test_combo_box_draws_display_text_and_stores_export_value():
    _, result = fill_by_name(pf.sample_form(), {"state": "TX"})
    reader = raw_fields(result.pdf)
    state = next(a.get_object() for a in reader.pages[0]["/Annots"] if a.get_object().get("/T") == "state")
    assert state["/V"] == "TX"
    assert b"Texas" in state["/AP"]["/N"].get_object().get_data()


@pytest.mark.parametrize(("value", "top", "shown", "hidden"), [
    ("Green", 0, [b"(Red)", b"(Green)"], [b"(Blue)"]),
    ("White", 3, [b"(Black)", b"(White)"], [b"(Red)", b"(Green)"]),  # scrolled to the selection
])
def test_list_box_shows_the_selection_highlighted(value, top, shown, hidden):
    form = pf.make_form([pf.Spec("listbox", "color", label="Color", y=360, width=120, height=40,
                                 options=["Red", "Green", "Blue", "Black", "White"], value="Red")])
    _, result = fill_by_name(form, {"color": value})
    reader = raw_fields(result.pdf)
    color = next(a.get_object() for a in reader.pages[0]["/Annots"] if a.get_object().get("/T") == "color")
    assert color["/V"] == value and color["/TI"] == top
    data = color["/AP"]["/N"].get_object().get_data()
    assert all(text in data for text in shown) and not any(text in data for text in hidden)
    assert b"re f" in data  # the selection is a filled rectangle
    text_objects = re.findall(rb"BT(.*?)ET", data, re.S)
    assert len(text_objects) == 2 and not any(b" re" in t for t in text_objects)  # no paths inside text
    assert reread(result.pdf)["color"].current_value == value


def test_unknown_or_read_only_fields_fail_without_blocking_others():
    form = open_form(pf.sample_form())
    ids = {f.name: f.id for f in form.info.fields}
    result = fill_pdf(form, {"F999": "x", ids["office_use"]: "x", ids["full_name"]: "Jane"})
    assert set(result.failed) == {"F999", ids["office_use"]}
    assert reread(result.pdf)["full_name"].current_value == "Jane"
    assert reread(result.pdf)["office_use"].current_value is None


def test_fallback_when_appearance_cannot_be_drawn():
    pdf = pf.set_field_keys(pf.sample_form(), "full_name", DA="0 g")  # no font: pypdf can't draw it
    _, result = fill_by_name(pdf, {"full_name": "Jane"})
    assert result.failed == {}
    assert any("appearance" in w for w in result.warnings)
    assert reread(result.pdf)["full_name"].current_value == "Jane"

    _, flat = fill_by_name(pdf, {"full_name": "Jane"}, flatten=True)
    assert list(flat.failed.values()) == ["The value could not be written into this field."]


def test_flatten_draws_values_and_removes_the_form():
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(pf.sample_form())))
    link = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Annot"),
            NameObject("/Subtype"): NameObject("/Link"),
            NameObject("/Rect"): ArrayObject([NumberObject(0), NumberObject(0), NumberObject(10), NumberObject(10)]),
        }
    )
    writer.pages[0]["/Annots"].append(writer._add_object(link))
    buf = io.BytesIO()
    writer.write(buf)

    _, result = fill_by_name(buf.getvalue(), {"full_name": "Jane Q. Public", "employment": "self"}, flatten=True)
    reader = raw_fields(result.pdf)
    assert "/AcroForm" not in reader.root_object
    annots = [a.get_object() for a in reader.pages[0].get("/Annots", [])]
    assert [a["/Subtype"] for a in annots] == ["/Link"]  # widgets gone, other annotations kept
    text = reader.pages[0].extract_text()
    assert "Jane Q. Public" in text and "Full name" in text
    xobjects = reader.pages[0]["/Resources"]["/XObject"]
    assert len([k for k in xobjects if k.startswith("/PdfWizardFm")]) == 11  # every widget drawn


def test_flatten_leaves_out_damaged_widgets_instead_of_failing():
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(pf.sample_form())))
    page = writer.pages[0]
    stray = DictionaryObject(  # a widget outside the field tree, with unreadable flags
        {
            NameObject("/Type"): NameObject("/Annot"),
            NameObject("/Subtype"): NameObject("/Widget"),
            NameObject("/F"): NameObject("/Bogus"),
            NameObject("/Rect"): ArrayObject([NumberObject(0), NumberObject(0), NumberObject(10), NumberObject(10)]),
        }
    )
    page["/Annots"].append(writer._add_object(stray))
    for ref in page["/Annots"]:
        if ref.get_object().get("/T") == "married":  # damage the checked appearance
            ref.get_object()["/AP"]["/N"]["/Yes"].get_object()[NameObject("/BBox")] = ArrayObject([NameObject("/a")] * 4)
    buf = io.BytesIO()
    writer.write(buf)

    form, result = fill_by_name(buf.getvalue(), {"full_name": "Jane Q. Public", "married": "Yes"}, flatten=True)
    assert result.failed == {by_name(form)["married"].id: "The value could not be drawn into the flattened PDF."}
    assert any(w.startswith("1 damaged form field(s)") for w in result.warnings)
    reader = raw_fields(result.pdf)
    assert "/AcroForm" not in reader.root_object and "/Annots" not in reader.pages[0]
    assert "Jane Q. Public" in reader.pages[0].extract_text()


def test_encrypted_form_is_re_encrypted_with_same_permissions():
    pdf = pf.encrypt_pdf(pf.sample_form(), permissions=4 | 256)
    _, result = fill_by_name(pdf, {"full_name": "Jane"})
    reader = raw_fields(result.pdf)
    assert reader.is_encrypted
    reader.decrypt("")
    assert int(reader.user_access_permissions) == 4 | 256
    assert reader.get_fields()["full_name"]["/V"] == "Jane"
    with pytest.raises(UnsupportedFormError, match="flattening"):
        fill_by_name(pdf, {"full_name": "Jane"}, flatten=True)


def test_hybrid_xfa_and_usage_rights_are_removed():
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(pf.add_xfa(pf.sample_form()))))
    writer.root_object[NameObject("/NeedsRendering")] = NameObject("/false")
    writer.root_object[NameObject("/Perms")] = DictionaryObject({NameObject("/UR3"): TextStringObject("sig")})
    buf = io.BytesIO()
    writer.write(buf)

    _, result = fill_by_name(buf.getvalue(), {"full_name": "Jane"})
    reader = raw_fields(result.pdf)
    assert "/XFA" not in reader.root_object["/AcroForm"]
    assert "/NeedsRendering" not in reader.root_object and "/Perms" not in reader.root_object
    assert any("Reader extended rights" in w for w in result.warnings)


# --------------------------------------------------------------------------- #
# Limits on hostile PDFs
# --------------------------------------------------------------------------- #

BIG = 9_000_000  # inflates past MAX_STREAM_BYTES, from about 9 KB in the file


def _counting(monkeypatch, owner, name: str) -> list[tuple]:
    """Replaces owner.name with a wrapper that records its arguments."""
    calls: list[tuple] = []
    real = getattr(owner, name)

    def wrapper(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, name, wrapper)
    return calls


def _full_scan_label(lines, rect, prefer_right: bool) -> str | None:
    """label_for without the baseline index: every line on the page is scored."""
    best = None
    for line in lines:
        score = inspector._label_score(line, *rect, prefer_right)
        if score is not None and (best is None or score < best[0]):
            best = (score, line.text)
    return best[1] if best else None


def _dense_text_pdf() -> bytes:
    """A crowded page: mixed font sizes, a large heading, baselines on .5 boundaries."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.setFont("Helvetica-Bold", 28)
    c.drawString(72, 740, "Application")
    for i in range(100):
        c.setFont("Helvetica", (6, 8, 10, 12, 18)[i % 5])
        c.drawString(40 + (i % 4) * 140, 700 - i * 6.5, f"Label {i}")
    c.showPage()
    c.save()
    return buf.getvalue()


@pytest.mark.parametrize("make_pdf", [pf.sample_form, _dense_text_pdf], ids=["form", "dense"])
def test_indexed_label_search_finds_the_same_labels_as_a_full_scan(monkeypatch, make_pdf):
    monkeypatch.setattr(inspector, "MAX_LABEL_LOOKUPS", 1_000_000)
    texts = inspector._PageTexts(PdfReader(io.BytesIO(make_pdf())))
    lines = texts.lines(1).lines
    assert len(lines) >= 10
    for x, y, (w, h) in itertools.product(range(0, 612, 41), range(0, 792, 29), [(12, 12), (120, 18), (220, 50)]):
        rect = (float(x), y + 0.5, float(x + w), y + 0.5 + h)
        for prefer_right in (False, True):
            assert texts.label_for(1, rect, prefer_right=prefer_right) == _full_scan_label(lines, rect, prefer_right)
        left = sorted((max(0.0, rect[0] - ln.x2), ln.text) for ln in lines if inspector._is_left_label(ln, *rect))
        assert texts.group_label_for(1, [rect], exclude=set()) == next((text for _, text in left), None)


@pytest.mark.parametrize(("states", "searches_per_field"), [(1, 2), (20, 2), (21, 1)])
def test_checkbox_label_searches_do_not_grow_with_states(monkeypatch, states, searches_per_field):
    calls = _counting(monkeypatch, inspector._PageTexts, "label_for")
    form = open_form(pf.make_state_heavy_form(fields=10, states=states, runs=50))
    assert [len(f.options) for f in form.info.fields] == [states] * 10
    # One search per widget (or none above MAX_LABELED_STATES) plus the field label; was states + 1.
    assert len(calls) == 10 * searches_per_field


def test_label_inference_work_is_capped(monkeypatch):
    monkeypatch.setattr(inspector, "MAX_LABEL_LOOKUPS", 40)
    monkeypatch.setattr(inspector, "MAX_LABEL_LINES", 100)
    scores = _counting(monkeypatch, inspector, "_label_score")
    # 120 checkboxes ask for 120 searches, and the huge heading puts every line
    # of the page inside every search band: uncapped, 120 x 301 lines are scored.
    pdf = pf.make_state_heavy_form(fields=120, states=1, runs=300, heading_size=500)
    form = open_form(pdf)
    assert len(form.info.fields) == 120
    assert 0 < len(scores) <= 40 * 100
    kept = inspector._PageTexts(PdfReader(io.BytesIO(pdf))).lines(1).lines
    assert len(kept) == 100 and kept[0].text == "H"  # the top of the page is kept


def test_labels_after_the_search_budget_are_left_empty(monkeypatch):
    monkeypatch.setattr(inspector, "MAX_LABEL_LOOKUPS", 3)
    fields = by_name(open_form(pf.sample_form()))
    labels = [fields[name].label for name in ("full_name", "dob", "ssn", "email", "notes")]
    assert labels == ["Full name", "Date of birth", "SSN", "Email address", None]  # email's is its tooltip


def test_pages_sharing_one_annots_array_are_read_once(monkeypatch):
    writer = PdfWriter()
    for _ in range(30):
        writer.add_blank_page(612, 792)
    widget = writer._add_object(DictionaryObject({
        NameObject("/Type"): NameObject("/Annot"),
        NameObject("/Subtype"): NameObject("/Widget"),
        NameObject("/FT"): NameObject("/Tx"),
        NameObject("/T"): TextStringObject("a"),
        NameObject("/Rect"): ArrayObject([NumberObject(10), NumberObject(10), NumberObject(100), NumberObject(30)]),
    }))
    shared = writer._add_object(ArrayObject([widget] * 1000))
    for page in writer.pages:
        page[NameObject("/Annots")] = shared
    writer.root_object[NameObject("/AcroForm")] = DictionaryObject({NameObject("/Fields"): ArrayObject([widget])})
    keys = _counting(monkeypatch, inspector, "_key")
    form = open_form(pf.to_bytes(writer))
    assert by_name(form)["a"].page == 1
    assert len(keys) < 2 * 1000  # was 30 x 1000


def test_stream_valued_entries_are_never_inflated():
    pdf = pf.set_field_stream(pf.sample_form(), "full_name", "V", b"0" * BIG)
    pdf = pf.set_field_stream(pdf, "dob", "TU", b"0" * BIG)
    form = open_form(pdf)
    fields = by_name(form)
    assert fields["full_name"].current_value is None
    assert fields["dob"].label == "Date of birth"  # the /TU stream is skipped; page text is used
    nodes = {ref.get_object()["/T"]: ref.get_object() for ref in form.reader.root_object["/AcroForm"]["/Fields"]}
    assert nodes["full_name"]["/V"].decoded_self is None  # pypdf never inflated (or cached) them
    assert nodes["dob"]["/TU"].decoded_self is None
    assert inspector._text(pf.flate_stream(b"an /Opt entry")) is None


def test_format_script_streams_are_inflated_with_a_bound():
    def dob_hint(pdf):
        return by_name(open_form(pdf))["dob"].format_hint

    assert dob_hint(pf.set_script_stream(pf.sample_form(), "dob", b'AFDate_FormatEx("mm/dd/yyyy");')) == "date:mm/dd/yyyy"
    assert dob_hint(pf.set_script_stream(pf.sample_form(), "dob", b"AFSpecial_Format(0);", compress=False)) == "zip"
    bomb = pf.set_script_stream(pf.sample_form(), "dob", b'AFDate_FormatEx("mm/dd/yyyy");' + b" " * BIG)
    form = open_form(bomb)
    assert by_name(form)["dob"].format_hint is None  # longer than MAX_SCRIPT_CHARS: no script
    dob = next(r.get_object() for r in form.reader.root_object["/AcroForm"]["/Fields"] if r.get_object()["/T"] == "dob")
    assert dob["/AA"]["/F"]["/JS"].decoded_self is None
    hexed = DecodedStreamObject()
    hexed.set_data(b"AFSpecial_Format(0);".hex().encode())
    hexed[NameObject("/Filter")] = NameObject("/ASCIIHexDecode")
    assert inspector._stream_script(hexed) == ""  # only plain and Flate scripts are read


def test_oversized_object_stream_fails_cleanly():
    assert open_form(pf.make_object_stream_bomb(100_000), for_filling=False).info.page_count == 1
    with pytest.raises(InvalidInputError, match="too large or complex") as caught:
        open_form(pf.make_object_stream_bomb(BIG), for_filling=False)
    assert isinstance(caught.value.__cause__, LimitReachedError)
    # pypdf keeps its limits in a ContextVar, so a new thread starts without them;
    # open_form must apply them itself.
    with ThreadPoolExecutor(1) as pool, pytest.raises(InvalidInputError, match="too large or complex"):
        pool.submit(open_form, pf.make_object_stream_bomb(BIG), for_filling=False).result()


def test_pypdf_caps_are_active_wherever_pypdf_decodes(monkeypatch):
    pdf = pf.encrypt_pdf(pf.sample_form())
    caps: list[int] = []

    def spy(real):
        def wrapper(*args, **kwargs):
            caps.append(pypdf.get_configuration().zlib_maximum_output_length)
            return real(*args, **kwargs)
        return wrapper

    monkeypatch.setattr(PageObject, "extract_text", spy(PageObject.extract_text))
    monkeypatch.setattr(PdfWriter, "write", spy(PdfWriter.write))
    form = open_form(pdf)  # label inference extracts page text
    form.plain_pdf()  # the decrypted copy, made later in the pipeline
    fill_pdf(form, {by_name(form)["full_name"].id: "Jane"})
    load_pdf_document(pdf, Context(sid="S1", name="doc.pdf", limits=Limits(), warnings=[]))  # adds the text layer
    assert len(caps) >= 5 and set(caps) == {MAX_STREAM_BYTES}
    assert pypdf.get_configuration().zlib_maximum_output_length > MAX_STREAM_BYTES  # nothing leaks out


@pytest.mark.parametrize(("error", "message"), [(LimitReachedError("x"), "too large or complex"),
                                                (ValueError("x"), "could not be read")])
def test_decrypted_copy_failures_are_user_errors(monkeypatch, error, message):
    form = open_form(pf.encrypt_pdf(pf.sample_form()))

    def broken(*args, **kwargs):
        raise error

    monkeypatch.setattr(inspector, "PdfWriter", broken)
    with pytest.raises(InvalidInputError, match=message):
        form.plain_pdf()
    monkeypatch.setattr(filler, "PdfWriter", broken)
    with pytest.raises(UnsupportedFormError, match="too large or complex" if message != "could not be read" else "damaged"):
        fill_pdf(form, {})


def test_pages_with_too_many_content_streams_are_skipped():
    def with_streams(count: int) -> bytes:
        writer = PdfWriter(clone_from=PdfReader(io.BytesIO(pf.make_text_pdf(["Hello"]))))
        page = writer.pages[0]
        extra = []
        for _ in range(count - 1):
            stream = DecodedStreamObject()
            stream.set_data(b"\n")
            extra.append(writer._add_object(stream))
        page[NameObject("/Contents")] = ArrayObject([page.raw_get("/Contents"), *extra])
        return pf.to_bytes(writer)

    assert "Hello" in page_text(PdfReader(io.BytesIO(with_streams(MAX_CONTENT_STREAMS))).pages[0])
    crowded = with_streams(MAX_CONTENT_STREAMS + 1)
    with pytest.raises(LimitReachedError):
        page_text(PdfReader(io.BytesIO(crowded)).pages[0])
    ctx = Context(sid="S1", name="doc.pdf", limits=Limits(), warnings=[])
    load_pdf_document(crowded, ctx)
    assert any("text could not be extracted" in w for w in ctx.warnings)


def test_encrypted_page_count_is_counted_not_declared():
    three = pf.make_form([pf.Spec("text", "a"), pf.Spec("text", "b", page=3)], pages=3)
    understated = pf.encrypt_with_page_count(three, 1)
    with pytest.raises(InvalidInputError, match="too many pages"):
        open_form(understated, limits=Limits(max_pages=2))
    form = open_form(understated)
    assert form.info.page_count == 3
    plain = PdfReader(io.BytesIO(form.plain_pdf()))
    assert len(plain.pages) == 3 and plain.root_object["/Pages"]["/Count"] == 3

    form = open_form(pf.encrypt_with_page_count(three, 50))  # used to be rejected as damaged
    assert form.info.page_count == 3 and by_name(form)["b"].page == 3
    filled = raw_fields(fill_pdf(form, {by_name(form)["b"].id: "x"}).pdf, password="")
    assert filled.root_object["/Pages"]["/Count"] == 3


@pytest.mark.parametrize("da", ["/F1\x000 12 Tf", "/Helv\x00q 12 Tf", "/H\u00e9lv 12 Tf", "/Helv \u0661\u0662 Tf",
                                "/Helv\x0b12 Tf"])
def test_list_box_font_from_da_must_be_plain_tokens(da):
    assert filler._DA_FONT.search(da) is None


@pytest.mark.parametrize("da", ["0\x00g", "0\u00a0g", "\u0660 g", "0\x0bg", "1 0 0\u2003rg"])
def test_list_box_color_from_da_must_be_plain_tokens(da):
    assert filler._DA_FILL.search(da) is None


def test_list_box_da_parsing_keeps_normal_values():
    assert filler._DA_FONT.search("/Helv 12 Tf 0 g").groups() == ("Helv", "12")
    assert filler._DA_FONT.search("/F#20x\t9.5\nTf").groups() == ("F#20x", "9.5")
    assert filler._DA_FILL.search("/Helv 12 Tf 0 g").group(0) == "0 g"
    assert filler._DA_FILL.search("/Helv 0 Tf .2 0.4 1 rg").group(0) == ".2 0.4 1 rg"
    assert filler._DA_FILL.search("/Helv 0 Tf 0 0 0 1 k").group(0) == "0 0 0 1 k"


def test_xfa_template_labels_are_extracted_and_xml_entities_rejected():
    from pypdf.generic import ArrayObject, DecodedStreamObject, DictionaryObject, NameObject, TextStringObject

    xml = """<template xmlns="http://www.xfa.org/schema/xfa-template/3.3/">
      <subform name="topmostSubform"><subform name="Page1">
        <field name="f1_01"><assist><speak>1. Name of entity. S S N or E I N.</speak></assist></field>
        <field name="f1_02"><caption><value><text>the treaty to claim a</text></value></caption>
          <assist><speak>10. of the treaty to claim a (enter percent).</speak></assist></field>
      </subform></subform>
    </template>"""
    stream = DecodedStreamObject()
    stream.set_data(xml.encode("utf-8"))
    acro = DictionaryObject({NameObject("/XFA"): ArrayObject([TextStringObject("template"), stream])})
    labels = inspector._xfa_labels(acro)
    assert labels["topmostSubform[0].Page1[0].f1_01[0]"] == "1. Name of entity. SSN or EIN"
    assert labels["topmostSubform[0].Page1[0].f1_02[0]"] == "10. of the treaty to claim a (enter percent)"

    # XML with <!ENTITY is ignored.
    bad_stream = DecodedStreamObject()
    bad_stream.set_data(b"<!DOCTYPE x [<!ENTITY a 'b'>]><template/>")
    bad_acro = DictionaryObject({NameObject("/XFA"): bad_stream})
    assert inspector._xfa_labels(bad_acro) == {}

    # Pure punctuation/symbol runs on a page never score as labels.
    assert inspector._label_score(inspector._Line(10, 10, 9, "▶"), 20, 8, 100, 20, False) is None

