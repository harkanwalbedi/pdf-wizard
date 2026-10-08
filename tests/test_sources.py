"""Tests for source loading (sources/): data files, documents, reference forms, and notes."""

from __future__ import annotations

import base64
import codecs
import functools
import io
import json
import logging
import re
import time
from datetime import date

import pytest
from pypdf import PdfReader

import pdf_factory as pf
from pdf_wizard.config import Limits
from pdf_wizard.errors import InvalidInputError, PdfWizardError, UnsupportedFormError
from pdf_wizard.models import Source, SourceKind
from pdf_wizard.sources import LoadedSources, SourceInput, emails, load_sources
from pdf_wizard.sources.common import escape_newlines

TODAY = date(2026, 10, 6)
CANARY = "987-65-4321"  # a fake SSN that must never show up in warnings, errors, or logs
NAME_CHARS = re.compile(r"[A-Za-z0-9 ._()\-]+")
UNSUPPORTED = "Unsupported document type for '{}'. Use PDF, PNG, JPEG, plain text, or .eml files."


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _bytes(content: str | bytes) -> bytes:
    return content.encode() if isinstance(content, str) else content


def data(content: str | bytes, name: str = "data.json", row: int | None = None) -> SourceInput:
    return SourceInput("data", name, _bytes(content), row=row)


def doc(content: str | bytes, name: str = "doc.txt") -> SourceInput:
    return SourceInput("doc", name, _bytes(content))


def ref(content: bytes, name: str = "ref.pdf") -> SourceInput:
    return SourceInput("reference", name, content)


def note(text: str | bytes) -> SourceInput:
    return SourceInput("note", "", _bytes(text))


def load(*inputs: SourceInput, limits: Limits | None = None) -> LoadedSources:
    return load_sources(inputs, limits=limits or Limits(), today=TODAY)


def load_one(item: SourceInput, limits: Limits | None = None) -> tuple[Source, list[str]]:
    """Loads a single input; returns (S1, warnings)."""
    loaded = load(item, limits=limits)
    assert [s.id for s in loaded.sources] == ["S0", "S1"]
    return loaded.sources[1], loaded.warnings


def error(*inputs: SourceInput, limits: Limits | None = None,
          exc: type[PdfWizardError] = InvalidInputError) -> str:
    """The user message of the error raised while loading `inputs`."""
    with pytest.raises(exc) as info:
        load(*inputs, limits=limits)
    return info.value.user_message


def filled_form() -> bytes:
    return pf.make_form(
        [
            pf.Spec("text", "full_name", label="Full name", y=700, value="Jane Doe"),
            pf.Spec("text", "ssn", label="SSN", y=670, width=120, value=CANARY),
            pf.Spec("text", "middle", label="Middle name", y=640),
            pf.Spec("text", "address", label="Address", y=610, value="1 Main St\nApt 2"),
            pf.Spec("checkbox", "married", label="Married", y=580, height=12, checked=True),
            pf.Spec("checkbox", "veteran", label="Veteran", y=550, height=12),
            pf.Spec("radio", "employment", label="Employed", value="employed", y=520, height=12),
            pf.Spec("radio", "employment", label="Self-employed", value="self", x=320, y=520, height=12,
                    checked=True),
            pf.Spec("choice", "state", label="State", y=490, width=120,
                    options=[("California", "CA"), ("Texas", "TX")], value="TX"),
        ]
    )


# --------------------------------------------------------------------------- #
# S0, IDs, caps
# --------------------------------------------------------------------------- #


def test_s0_states_todays_date_and_comes_first():
    loaded = load()
    assert len(loaded.sources) == 1 and loaded.warnings == []
    s0 = loaded.sources[0]
    assert (s0.id, s0.kind, s0.name) == ("S0", SourceKind.SYSTEM, "PDF Wizard")
    assert s0.text == "Today's date: 2026-10-06 (Tuesday, October 6, 2026)."
    other = load_sources([], limits=Limits(), today=date(2027, 1, 3)).sources[0]
    assert other.text == "Today's date: 2027-01-03 (Sunday, January 3, 2027)."


def test_ids_follow_input_order_and_skipped_notes_use_no_id():
    loaded = load(note("first"), data('{"a": 1}'), note("  \n "), doc("hello"), note("last"))
    assert [(s.id, s.kind, s.name) for s in loaded.sources[1:]] == [
        ("S1", SourceKind.NOTE, "Note 1"),
        ("S2", SourceKind.STRUCTURED, "data.json"),
        ("S3", SourceKind.TEXT, "doc.txt"),
        ("S4", SourceKind.NOTE, "Note 3"),  # named by position among the notes given
    ]


def test_file_size_and_count_caps():
    assert error(doc(b"a" * 101, name="big.txt"), limits=Limits(max_file_bytes=100)) == (
        "'big.txt' is too large (limit 100 bytes)."
    )
    assert error(doc(b"a" * (Limits().max_file_bytes + 1), name="big.txt")) == (
        "'big.txt' is too large (limit 10 MB)."
    )
    assert error(doc(b"", name="empty.txt")) == "'empty.txt' is empty."
    three = [doc("a", name=f"{i}.txt") for i in range(3)]
    assert error(*three, limits=Limits(max_source_files=2)) == "Too many files (limit 2)."
    # Notes don't count as files.
    assert len(load(*three[:2], *[note("n")] * 10, limits=Limits(max_source_files=2)).sources) == 13
    assert error(*[note("n")] * 11) == "Too many notes (limit 10)."
    assert error(doc(b"a" * 60), doc(b"b" * 60), limits=Limits(max_total_bytes=100)) == (
        "The files are too large in total (limit 100 bytes)."
    )


def test_unknown_role_is_a_programming_error():
    with pytest.raises(ValueError):
        load(SourceInput("other", "x", b"x"))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/passwd", "passwd"),
        ("C:\\Users\\jane\\W-2 2025 (final).txt", "W-2 2025 (final).txt"),
        ("bad\x00\x1b[31mname\n.txt", "bad_31mname.txt"),
        ("résumé.txt", "resume.txt"),
        ("简历.txt", "_.txt"),
        ('a"b<c>|d;e.txt', "a_b_c_d_e.txt"),
        ("", "file"),
        ("dir/", "file"),
        ("...", "file"),
        ("a" * 200 + ".txt", "a" * 76 + ".txt"),
        ("b" * 200, "b" * 80),
    ],
)
def test_display_names_are_sanitized(raw: str, expected: str):
    source, _ = load_one(doc("hello", name=raw))
    assert source.name == expected
    assert NAME_CHARS.fullmatch(source.name) and len(source.name) <= 80


# --------------------------------------------------------------------------- #
# Structured data: JSON
# --------------------------------------------------------------------------- #


def test_json_record_is_flattened_to_key_paths():
    record = """{
      "applicant": {
        "name": " Jane Doe ", "married": true, "veteran": false, "middle": null, "suffix": "",
        "children": [{"name": "Ann", "age": 7}, {"name": "Bob", "pets": []}],
        "income": 1.50, "big": 123456789012345678901234567890, "rate": 1E-5, "debt": -0.0
      },
      "tags": ["a", "b"],
      "address": "1 Main St\\nApt 2",
      "a.b": "dotted key"
    }"""
    source, warnings = load_one(data(record))
    assert source.kind is SourceKind.STRUCTURED and source.name == "data.json"
    assert source.keypaths == {
        "applicant.name": "Jane Doe",
        "applicant.married": "true",
        "applicant.veteran": "false",
        "applicant.children[0].name": "Ann",
        "applicant.children[0].age": "7",
        "applicant.children[1].name": "Bob",
        "applicant.income": "1.50",  # numbers keep their exact text
        "applicant.big": "123456789012345678901234567890",
        "applicant.rate": "1E-5",
        "applicant.debt": "-0.0",
        "tags[0]": "a",
        "tags[1]": "b",
        "address": "1 Main St\nApt 2",
        '["a.b"]': "dotted key",  # quoted, so it can't be confused with {"a": {"b": ...}}
    }
    lines = source.text.split("\n")
    assert lines[0] == "applicant.name = Jane Doe"
    assert "address = 1 Main St\\nApt 2" in lines  # newlines are escaped in the text only
    assert len(lines) == len(source.keypaths)
    assert source.data is None and source.verifiable and warnings == []


def test_json_with_bom_and_leading_whitespace():
    assert load_one(data('\ufeff \n {"a": "x"}'))[0].keypaths == {"a": "x"}


def test_json_rejects_nan_and_malformed_documents():
    assert error(data('{"a": NaN}')) == "'data.json' is not valid JSON."
    assert error(data('{"a": -Infinity}')) == "'data.json' is not valid JSON."
    message = error(data(f'{{"ssn": "{CANARY}",}}'))
    assert re.fullmatch(r"'data\.json' is not valid JSON \(line 1, column \d+\)\.", message)
    assert error(data("[1, 2]")) == "'data.json' must contain a JSON object or a list of objects."
    assert error(data("[]")) == "'data.json' has no records."
    assert error(data('{"a": null, "b": ""}')) == "'data.json' has no values in the selected record."


def test_json_array_needs_a_row_when_it_has_several_records():
    people = '[{"name": "Ann"}, {"name": "Bob"}, {"name": "Cy"}]'
    assert error(data(people)) == "'data.json' has 3 records; choose one with --row (1-3)."
    assert load_one(data(people, row=2))[0].keypaths == {"name": "Bob"}
    assert error(data(people, row=4)) == "Row 4 is out of range: 'data.json' has 3 records."
    assert error(data(people, row=0)) == "Row numbers start at 1."


def test_single_record_files_ignore_the_row_number():
    # The CLI and the web app pass one row number for every data file.
    loaded = load(
        data('[{"name": "Ann"}]'),
        data('[{"name": "Ann"}]', row=2),
        data('{"name": "Ann"}', row=1),
        data('{"name": "Ann"}', row=3),
        data("name\nAnn\n", name="one.csv", row=2),
        data("name,city\nAnn,Oslo\nBob,Rome\n", name="people.csv", row=2),
    )
    assert [s.keypaths for s in loaded.sources[1:]] == [{"name": "Ann"}] * 5 + [{"name": "Bob", "city": "Rome"}]
    assert loaded.warnings == [
        "S2: only one record; the row number was not used.",
        "S4: only one record; the row number was not used.",
        "S5: only one record; the row number was not used.",
    ]


def test_json_depth_cap():
    limits = Limits(max_json_depth=3)
    too_deep = "'data.json' is nested too deeply (limit 3 levels)."
    assert load_one(data('{"a": {"b": {"c": 1}}}'), limits)[0].keypaths == {"a.b.c": "1"}
    assert error(data('{"a": {"b": {"c": {"d": 1}}}}'), limits=limits) == too_deep
    assert error(data('[{"a": {"b": {"c": 1}}}]'), limits=limits) == too_deep  # the array is a level
    deep = '{"a": ' * 40 + "1" + "}" * 40
    assert error(data(deep)) == "'data.json' is nested too deeply (limit 32 levels)."
    assert error(data("[" * 100_000)) == "'data.json' is nested too deeply."  # parser RecursionError


def test_json_key_cap():
    limits = Limits(max_json_keys=5)
    five = '{"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}'
    assert len(load_one(data(five), limits)[0].keypaths) == 5
    assert error(data('{"a": [1, 2, 3, 4, 5]}'), limits=limits) == "'data.json' has too many keys (limit 5)."


def test_structured_text_is_never_truncated():
    assert error(data('{"name": "' + "x" * 30 + '"}'), limits=Limits(max_source_chars=20)) == (
        "'data.json' has too much data (limit 20 characters); remove unused fields."
    )


# --------------------------------------------------------------------------- #
# Structured data: CSV
# --------------------------------------------------------------------------- #


def test_csv_with_bom_and_one_row():
    content = "\ufeffName, DOB ,Notes\r\nJane Doe,1990-01-02,\r\n".encode()
    source, warnings = load_one(data(content, name="person.csv"))
    assert source.kind is SourceKind.STRUCTURED and warnings == []
    assert source.keypaths == {"Name": "Jane Doe", "DOB": "1990-01-02"}  # names stripped, empties skipped
    assert source.text == "Name = Jane Doe\nDOB = 1990-01-02"


def test_csv_semicolon_delimiter():
    source, _ = load_one(data("name;city;amount\nJane;Paris;1,50\n", name="eu.csv"))
    assert source.keypaths == {"name": "Jane", "city": "Paris", "amount": "1,50"}


def test_csv_quoted_values():
    source, _ = load_one(data('name,address\n"Doe, Jane","1 Main St\nApt 2"\n', name="d.csv"))
    assert source.keypaths == {"name": "Doe, Jane", "address": "1 Main St\nApt 2"}
    assert source.text == "name = Doe, Jane\naddress = 1 Main St\\nApt 2"


def test_csv_with_several_rows_needs_a_row():
    people = "name,city\nAnn,Oslo\nBob,Rome\n\n,\n"  # trailing blank and empty rows aren't records
    assert error(data(people, name="people.csv")) == "'people.csv' has 2 records; choose one with --row (1-2)."
    assert load_one(data(people, name="people.csv", row=2))[0].keypaths == {"name": "Bob", "city": "Rome"}
    assert error(data(people, name="people.csv", row=3)) == "Row 3 is out of range: 'people.csv' has 2 records."


def test_csv_header_problems():
    assert error(data("name, name\nA,B\n", name="d.csv")) == "'d.csv' has duplicate column names in its header row."
    assert error(data("name,,city\nA,B,C\n", name="d.csv")) == "'d.csv' has an empty column name in its header row."
    assert error(data("name,city\n", name="d.csv")) == "'d.csv' has no data rows."
    assert error(data("a,b\n1,2,3\n", name="d.csv")) == (
        "Row 1 of 'd.csv' has more values than there are columns in the header."
    )
    assert error(data("a,b,c,d,e,f\n1,2,3,4,5,6\n", name="d.csv"), limits=Limits(max_json_keys=5)) == (
        "'d.csv' has too many columns (limit 5)."
    )


def test_data_files_must_be_utf8_json_or_csv():
    assert error(data("name\ncafé\n".encode("latin-1"), name="d.csv")) == "'d.csv' is not valid UTF-8 text."
    assert error(data("name\nJane\n".encode("utf-16"), name="d.csv")) == "'d.csv' is not a JSON or CSV file."
    assert error(data(pf.make_png(), name="d.csv")) == "'d.csv' is not a JSON or CSV file."
    assert error(data(pf.make_text_pdf(["x"]), name="d.json")) == "'d.json' is not a JSON or CSV file."


# --------------------------------------------------------------------------- #
# Documents: text, images, type detection
# --------------------------------------------------------------------------- #


def test_plain_text_document():
    content = "\ufeffDear Jane,\r\nYour salary is 90,000.\rRegards\n\n".encode()
    source, warnings = load_one(doc(content, name="letter.txt"))
    assert (source.kind, source.name) == (SourceKind.TEXT, "letter.txt")
    assert source.text == "Dear Jane,\nYour salary is 90,000.\nRegards"
    assert source.data is None and warnings == []


def test_text_document_rejections():
    assert error(doc(b"hello\x00world", name="notes.txt")) == UNSUPPORTED.format("notes.txt")
    assert error(doc("café".encode("latin-1"), name="notes.txt")) == (
        "'notes.txt' is not UTF-8 text; save it as UTF-8 and try again."
    )
    assert error(doc(" \n\t ", name="blank.txt")) == "'blank.txt' has no text."
    # NUL bytes rule out email too, whatever the name.
    nul_email = b"From: a@b.example\nSubject: hi\nMIME-Version: 1.0\n\nhi\x00"
    assert error(doc(nul_email, name="x.eml")) == UNSUPPORTED.format("x.eml")


def test_long_text_is_truncated_with_a_warning():
    source, warnings = load_one(doc("x" * 25), Limits(max_source_chars=10))
    assert source.text == "x" * 10
    assert warnings == ["S1 was truncated to 10 characters."]


def test_document_type_comes_from_content_not_extension():
    png, jpeg = pf.make_png(), pf.make_jpeg()
    loaded = load(
        doc(png, name="form.pdf"),
        doc(jpeg, name="scan.txt"),
        doc("just text", name="photo.png"),
        doc(pf.make_text_pdf(["Employer: Acme"]), name="notes.txt"),
    )
    image, photo, text, pdf = loaded.sources[1:]
    assert (image.kind, image.media_type, image.data, image.text) == (SourceKind.IMAGE, "image/png", png, None)
    assert not image.verifiable  # code can't check evidence against an image
    assert (photo.kind, photo.media_type, photo.data) == (SourceKind.IMAGE, "image/jpeg", jpeg)
    assert (text.kind, text.text) == (SourceKind.TEXT, "just text")
    assert pdf.kind is SourceKind.PDF
    assert error(doc(bytes(range(256)) * 4, name="x.txt")) == UNSUPPORTED.format("x.txt")


# --------------------------------------------------------------------------- #
# Documents: email
# --------------------------------------------------------------------------- #

PLAIN_EMAIL = (
    "From: HR Team <hr@acme.example>\r\n"
    "To: jane@example.com\r\n"
    "Subject: =?utf-8?q?Caf=C3=A9_offer?=\r\n"
    "Date: Mon, 5 Oct 2026 10:00:00 +0000\r\n"
    "Message-ID: <1@acme.example>\r\n"
    "Content-Type: text/plain; charset=utf-8\r\n"
    "\r\n"
    "Your salary is $90,000.\r\n"
)

HTML_EMAIL = (
    "From: payroll@acme.example\n"
    "Subject: Pay stub\n"
    "MIME-Version: 1.0\n"
    "Content-Type: text/html; charset=utf-8\n"
    "\n"
    "<html><head><style>p { color: red }</style><script>var secret = 'x';</script></head>"
    "<body><p>Employer: <b>Acme</b></p><div>Wages:&nbsp;50,000</div>"
    "<table><tr><td>Name</td><td>Jane</td></tr></table>Thanks<br>HR<!-- hidden comment --></body></html>\n"
)

ATTACHMENT_EMAIL = (
    "From: hr@acme.example\n"
    "Subject: Your documents\n"
    "MIME-Version: 1.0\n"
    'Content-Type: multipart/mixed; boundary="XYZ"\n'
    "\n"
    "--XYZ\n"
    'Content-Type: multipart/alternative; boundary="ALT"\n'
    "\n"
    "--ALT\n"
    "Content-Type: text/plain; charset=utf-8\n"
    "\n"
    "See the attached W-2.\n"
    "--ALT\n"
    "Content-Type: text/html; charset=utf-8\n"
    "\n"
    "<p>See the <b>attached</b> W-2.</p>\n"
    "--ALT--\n"
    "--XYZ\n"
    "Content-Type: application/pdf\n"
    f'Content-Disposition: attachment; filename="w2-{CANARY}.pdf"\n'
    "Content-Transfer-Encoding: base64\n"
    "\n"
    "JVBERi0xLjQK\n"
    "--XYZ--\n"
)


@pytest.mark.parametrize("name", ["offer.eml", "offer.txt"])  # detected by name or by content
def test_plain_text_email(name: str):
    source, warnings = load_one(doc(PLAIN_EMAIL, name=name))
    assert source.kind is SourceKind.EMAIL and warnings == []
    assert source.text == (
        "From: HR Team <hr@acme.example>\n"
        "To: jane@example.com\n"
        "Date: Mon, 5 Oct 2026 10:00:00 +0000\n"
        "Subject: Café offer\n"
        "\n"
        "Your salary is $90,000."
    )


def test_html_only_email_is_converted_to_text():
    source, _ = load_one(doc(HTML_EMAIL, name="stub.eml"))
    assert source.kind is SourceKind.EMAIL
    assert source.text == (
        "From: payroll@acme.example\nSubject: Pay stub\n\n"
        "Employer: Acme\n\nWages: 50,000\n\nName Jane\n\nThanks\nHR"
    )
    for leftover in ("<", ">", "secret", "color", "&nbsp;", "hidden"):
        assert leftover not in source.text


def test_email_attachments_are_ignored_with_a_warning():
    source, warnings = load_one(doc(ATTACHMENT_EMAIL, name="docs.eml"))
    assert source.text == "From: hr@acme.example\nSubject: Your documents\n\nSee the attached W-2."
    assert warnings == ["S1: email attachments were ignored; upload them as separate documents."]


def test_memo_with_email_like_headers_stays_plain_text():
    memo = "To: All staff\nFrom: Jane Doe\nSubject: Parking\n\nEmployee ID: 12345\nSent: Monday\n"
    source, _ = load_one(doc(memo, name="memo.txt"))
    assert source.kind is SourceKind.TEXT and source.text == memo.strip()


def test_eml_without_headers_keeps_its_text():
    source, _ = load_one(doc("Hello Jane,\nYour salary is 90,000.\n", name="note.eml"))
    assert source.kind is SourceKind.EMAIL and source.text == "Hello Jane,\nYour salary is 90,000."


def test_hostile_emails_are_rejected_cheaply():
    too_complex = "'x.eml' has too many parts or headers to be read as an email."
    long_type = "From: a@b.example\nContent-Type: text/plain; " + "(a)" * 2000 + "\n\nhi\n"
    assert error(doc(long_type, name="x.eml")) == too_complex
    parts = "".join("--B\nContent-Type: text/plain\n\nx\n" for _ in range(2001))
    many_parts = f'From: a@b.example\nContent-Type: multipart/mixed; boundary="B"\n\n{parts}--B--\n'
    assert error(doc(many_parts, name="x.eml")) == too_complex


def test_huge_address_headers_are_capped_not_parsed():
    hostile = "From: " + "a," * 50_000 + "\nSubject: hi\nMIME-Version: 1.0\n\nbody\n"
    source, _ = load_one(doc(hostile, name="x.eml"))
    from_line, subject_line = source.text.split("\n")[:2]
    assert from_line.startswith("From: a,a,a,") and len(from_line) == len("From: ") + 2000
    assert subject_line == "Subject: hi" and source.text.endswith("\n\nbody")


def email_with(charset: str, body: bytes, *, subject: str = "Hi", cte: str | None = None) -> bytes:
    head = f"From: a@b.example\nSubject: {subject}\nMIME-Version: 1.0\nContent-Type: text/plain; charset={charset}\n"
    if cte:
        head += f"Content-Transfer-Encoding: {cte}\n"
    return head.encode() + b"\n" + body


@pytest.mark.parametrize(
    ("charset", "body", "expected"),
    [
        ("iso-8859-1", "café".encode("latin-1"), "café"),
        ("latin1", "café".encode("latin-1"), "café"),
        ("windows-1252", b"\x80 5", "€ 5"),
        ("shift_jis", "山田".encode("shift_jis"), "山田"),
        ("windows-31j", "山田".encode("cp932"), "山田"),
        ("iso-2022-jp", "山田".encode("iso2022_jp"), "山田"),
        ("ks_c_5601-1987", "김".encode("euc_kr"), "김"),
        ("gb2312", "王".encode("gb2312"), "王"),
        ("big5", "王".encode("big5"), "王"),
        ("koi8-r", "Иван".encode("koi8_r"), "Иван"),
        ("macintosh", "café".encode("mac_roman"), "café"),
    ],
)
def test_email_bodies_use_allowed_charsets(charset: str, body: bytes, expected: str):
    source, _ = load_one(doc(email_with(charset, body), name="m.eml"))
    assert source.text.split("\n\n", 1)[1] == expected


def test_utf16_email_body():
    body = base64.b64encode("café".encode("utf-16"))
    source, _ = load_one(doc(email_with("UTF-16", body, cte="base64"), name="m.eml"))
    assert source.text.split("\n\n", 1)[1] == "café"


@pytest.mark.parametrize(
    ("charset", "body"),
    [
        ("punycode", b"abc-"),  # punycode would give "abc", with a quadratic decoder
        ("idna", b"xn--caf-dma"),  # idna would give "café"
        ("utf-7", b"+AGEAYgBj-"),  # utf-7 would give "abc"
        ("unicode_escape", b"\\u0041"),
        ("base64", b"YWJj"),
        ("x-unknown", b"plain"),
        ('"' + "a" * 41 + '"', b"plain"),  # longer than any charset name
    ],
)
def test_other_email_charsets_are_read_as_utf8(charset: str, body: bytes):
    source, _ = load_one(doc(email_with(charset, body), name="m.eml"))
    assert source.text.split("\n\n", 1)[1] == body.decode()


def test_email_header_words_use_allowed_charsets_only():
    subject = "=?iso-8859-1?q?caf=E9?= =?punycode?q?abc-?= =?idna?q?xn--caf-dma?= =?iso-8859-1*fr?q?=E9t=E9?="
    source, _ = load_one(doc(email_with("utf-8", b"body", subject=subject), name="m.eml"))
    assert source.text.split("\n")[1] == "Subject: caféabc-xn--caf-dmaété"
    source, _ = load_one(doc(email_with("utf-8", b"body", subject="Re: Café =?utf-8?q?D=C3=B6e?= ok"), name="m.eml"))
    assert source.text.split("\n")[1] == "Subject: Re: Café Döe ok"


def test_large_punycode_email_is_read_quickly():
    mail = email_with("punycode", b"a" * 1_000_000, subject="=?punycode?q?" + "a" * 3000 + "?=")
    start = time.perf_counter()
    source, _ = load_one(doc(mail, name="m.eml"))
    assert time.perf_counter() - start < 5  # the punycode decoder needs about 15 s for this body
    assert source.text.endswith("a" * 100)


@pytest.mark.parametrize(
    ("charset", "codec"),
    [
        ("UTF-8", "utf-8"),
        ("us-ascii", "ascii"),
        ("Latin-1", "iso8859-1"),
        ("ISO_8859-1:1987", "iso8859-1"),
        ("iso-8859-15", "iso8859-15"),
        ("Windows-1258", "cp1258"),
        ("utf-16le", "utf-16-le"),
        ("cp936", "gbk"),
        ("big5-hkscs", "big5hkscs"),
        ("sjis", "shift_jis"),
        ("windows-31j", "cp932"),
        ("euc-jp", "euc_jp"),
        ("ks_c_5601-1987", "euc_kr"),
        ("ISO-2022-JP-2", "iso2022_jp_2"),
        ("koi8-u", "koi8-u"),
        ("macintosh", "mac-roman"),
        ("punycode", "utf-8"),
        ("IDNA", "utf-8"),
        ("utf-7", "utf-8"),
        ("raw_unicode_escape", "utf-8"),
        ("rot13", "utf-8"),
        ("zlib", "utf-8"),
        ("cp037", "utf-8"),  # EBCDIC: a real text codec, but not on the list
        ("x-unknown", "utf-8"),
        ("", "utf-8"),
        (None, "utf-8"),
    ],
)
def test_text_codec_resolves_aliases_within_the_allow_list(charset: str | None, codec: str):
    assert emails.text_codec(charset) == codec


def test_unknown_charsets_never_reach_the_codec_registry(monkeypatch: pytest.MonkeyPatch):
    looked_up: list[str] = []
    real_lookup = codecs.lookup

    def lookup(name: str) -> codecs.CodecInfo:
        looked_up.append(name)
        return real_lookup(name)

    monkeypatch.setattr(codecs, "lookup", lookup)
    for charset in ("x-bogus-1", "x-bogus-2", "punycode", "idna", "utf-7", "Latin1"):
        emails.text_codec(charset)
    assert looked_up == ["latin_1"]  # a miss costs an attempt to import a codec module


def test_repeated_headers_that_decode_to_nothing_are_capped(monkeypatch: pytest.MonkeyPatch):
    decoded: list[str] = []
    real_decode = emails._decode_header

    def decode(raw: str) -> str:
        decoded.append(raw)
        return real_decode(raw)

    monkeypatch.setattr(emails, "_decode_header", decode)
    empty_words = " ".join(f"=?x-{i}?q??=" for i in range(200))  # unknown charsets, no text
    mail = b"From: a@b.example\nMIME-Version: 1.0\n" + f"Subject: {empty_words}\n".encode() * 1000 + b"\nbody\n"
    source, _ = load_one(doc(mail, name="m.eml"))
    assert source.text == "From: a@b.example\n\nbody"
    assert len(decoded) == 2  # From and the first Subject, not all 1,000 Subject headers


# --------------------------------------------------------------------------- #
# One line per value: "key = value" and field-value lines can't be forged
# --------------------------------------------------------------------------- #


@functools.cache
def line_breaks() -> list[str]:
    """Every character that str.splitlines() treats as a line boundary."""
    return [chr(i) for i in range(0x110000) if len(f"a{chr(i)}b".splitlines()) > 1]


def test_escape_newlines_covers_every_splitlines_boundary():
    assert line_breaks() == ["\n", "\v", "\f", "\r", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"]
    for ch in line_breaks():
        escaped = escape_newlines(f"a{ch}b")
        assert escaped.splitlines() == [escaped] and escaped.isprintable()
    assert escape_newlines("a\r\nb\rc\nd") == "a\\nb\\nc\\nd"
    assert escape_newlines("1\v2\f3\x1c4\x1d5\x1e6\x857\u20288\u20299") == (
        "1\\v2\\f3\\x1c4\\x1d5\\x1e6\\x857\\u20288\\u20299"
    )


def test_json_keys_and_values_stay_on_one_line():
    record = {"name": "Jane\u2028ssn = 999-99-9999", "city": "Oslo\u2029zip = 1", "a\u2028b": "x", "note": "a\r\nb"}
    source, _ = load_one(data(json.dumps(record)))
    assert source.text.splitlines() == [
        "name = Jane\\u2028ssn = 999-99-9999",
        "city = Oslo\\u2029zip = 1",
        '["a\\u2028b"] = x',
        "note = a\\nb",
    ]
    assert source.keypaths["name"] == "Jane\u2028ssn = 999-99-9999"  # raw values are kept for checks


def test_csv_keys_and_values_stay_on_one_line():
    source, _ = load_one(data("first\u2028name,note\nJane,ok\u2029ssn = 1\n", name="d.csv"))
    assert source.keypaths == {"first name": "Jane", "note": "ok\u2029ssn = 1"}
    assert source.text.splitlines() == ["first name = Jane", "note = ok\\u2029ssn = 1"]


def test_reference_form_lines_stay_on_one_line():
    pdf = pf.make_form([pf.Spec("text", "full_name", label="Full name", value="Jane")])
    pdf = pf.set_field_keys(pdf, "full_name", T="full\u2028name", TU="Name\u2029x", V='Jane\u2028"SSN" [ssn] = 999')
    source, _ = load_one(ref(pdf))
    lines = source.text.splitlines()
    assert len(lines) == 2  # the header and one field
    assert lines[1].endswith('[full\\u2028name] = Jane\\u2028"SSN" [ssn] = 999')


def test_email_headers_stay_on_one_line():
    subject = base64.b64encode("Hi\u2028From: forged\u2029Cc: x".encode()).decode()
    source, _ = load_one(doc(email_with("utf-8", b"body", subject=f"=?utf-8?b?{subject}?="), name="m.eml"))
    assert source.text.splitlines() == ["From: a@b.example", "Subject: Hi From: forged Cc: x", "", "body"]


# --------------------------------------------------------------------------- #
# Documents: PDF
# --------------------------------------------------------------------------- #


def test_pdf_document_sends_the_file_and_keeps_its_text_layer_for_checks():
    pdf = pf.make_text_pdf(["Employer: Acme Corp", "Wages: 50,000"], pages=2)
    source, warnings = load_one(doc(pdf, name="w2.pdf"))
    assert (source.kind, source.media_type, source.data) == (SourceKind.PDF, "application/pdf", pdf)
    assert source.text is None  # the model reads the file itself
    page = "Employer: Acme Corp\nWages: 50,000"
    assert source.check_text == f"[Page 1]\n{page}\n\n[Page 2]\n{page}"
    assert source.verifiable and warnings == []


def test_pdf_text_layer_is_capped():
    pdf = pf.make_text_pdf(["Employer: Acme Corp", "Wages: 50,000"], pages=2)
    source, warnings = load_one(doc(pdf, name="w2.pdf"), Limits(max_source_chars=30))
    assert source.check_text == "[Page 1]\nEmployer: Acme Corp\nWages: 50,000"[:30]
    assert warnings == ["S1: only the first 30 characters of its text are used to check evidence."]


def test_scanned_pdf_has_no_text_and_is_unverifiable():
    pdf = pf.make_blank_pdf(2)
    source, warnings = load_one(doc(pdf, name="scan.pdf"))
    assert source.kind is SourceKind.PDF and source.data == pdf
    assert source.text is None and source.check_text is None and not source.verifiable
    assert warnings == []


def test_filled_pdf_document_includes_its_field_values():
    source, _ = load_one(doc(filled_form(), name="old.pdf"))
    lines = source.text.split("\n")
    assert source.kind is SourceKind.PDF
    assert lines[0] == "Form field values in this PDF:"
    assert '"Full name" [full_name] = Jane Doe' in lines
    assert source.check_text.startswith("[Page 1]\n") and source.check_text.endswith("\n\n" + source.text)


# --------------------------------------------------------------------------- #
# Reference forms
# --------------------------------------------------------------------------- #


def test_reference_form_values_become_text():
    pdf = filled_form()
    source, warnings = load_one(ref(pdf, name="last-year.pdf"))
    assert (source.kind, source.name, source.media_type) == (
        SourceKind.REFERENCE_FORM, "last-year.pdf", "application/pdf"
    )
    assert source.data == pdf
    assert source.text.split("\n") == [
        "Field values from a previously filled form:",
        '"Full name" [full_name] = Jane Doe',
        f'"SSN" [ssn] = {CANARY}',
        '"Address" [address] = 1 Main St\\nApt 2',  # newline escaped
        '"Married" [married] = Yes (Married)',  # unchecked "veteran" and empty "middle" are skipped
        '"employment" [employment] = self (Self-employed)',
        '"State" [state] = TX (Texas)',
    ]
    assert source.check_text.startswith(source.text + "\n\n[Page 1]\n")
    assert "Veteran" in source.check_text  # printed labels come from the page text layer
    assert warnings == []


def test_reference_without_values_falls_back_to_page_text():
    blank = pf.make_form([pf.Spec("text", "full_name", label="Full name")])
    flat = pf.make_text_pdf(["Full name: Jane Doe"])  # e.g. a flattened copy
    loaded = load(ref(blank, name="blank.pdf"), ref(flat, name="flat.pdf"))
    blank_src, flat_src = loaded.sources[1:]
    assert blank_src.text is None and blank_src.check_text == "[Page 1]\nFull name"
    assert flat_src.text is None and flat_src.check_text == "[Page 1]\nFull name: Jane Doe"
    assert (flat_src.kind, flat_src.data) == (SourceKind.REFERENCE_FORM, flat)
    assert loaded.warnings == [
        "S1: no filled fields found; using the page text only.",
        "S2: no filled fields found; using the page text only.",
    ]


def test_password_protected_pdfs_are_rejected():
    locked = pf.encrypt_pdf(filled_form(), user_password="s3cret")
    expected = "'locked.pdf': This PDF is password-protected. Remove the password and try again."
    assert error(ref(locked, name="locked.pdf"), exc=UnsupportedFormError) == expected
    assert error(doc(locked, name="locked.pdf"), exc=UnsupportedFormError) == expected


def test_owner_password_only_pdf_is_decrypted_for_the_model():
    source, _ = load_one(ref(pf.encrypt_pdf(filled_form())))  # empty user password
    assert '"Full name" [full_name] = Jane Doe' in source.text.split("\n")
    assert not PdfReader(io.BytesIO(source.data)).is_encrypted


def test_reference_must_be_a_readable_pdf():
    assert error(ref(pf.make_png(), name="ref.pdf")) == "'ref.pdf' is not a PDF; reference forms must be PDF files."
    assert error(ref(b"Full name: Jane", name="ref.txt")) == (
        "'ref.txt' is not a PDF; reference forms must be PDF files."
    )
    assert error(ref(b"%PDF-1.7\nnot really a pdf\n", name="bad.pdf")).startswith("'bad.pdf': ")
    assert error(doc(b"%PDF-1.7\nnot really a pdf\n", name="bad.pdf")).startswith("'bad.pdf': ")


# --------------------------------------------------------------------------- #
# Notes
# --------------------------------------------------------------------------- #


def test_notes():
    limits = Limits(max_note_chars=10)
    loaded = load(note(""), note("  a\x00b\tc\r\nd\x1b  "), note(" " * 5 + "x" * 10 + " " * 5), limits=limits)
    assert [(s.id, s.kind, s.name, s.text) for s in loaded.sources[1:]] == [
        ("S1", SourceKind.NOTE, "Note 2", "ab\tc\nd"),  # control characters dropped
        ("S2", SourceKind.NOTE, "Note 3", "x" * 10),  # length counted after stripping
    ]
    assert error(note("x" * 11), limits=limits) == "Note 1 is too long (limit 10 characters)."
    assert error(note(b"caf\xe9")) == "Note 1 is not valid UTF-8 text."


# --------------------------------------------------------------------------- #
# No personal data in warnings, errors, logs, or reprs
# --------------------------------------------------------------------------- #


def test_warnings_never_contain_values_or_file_names():
    email = ATTACHMENT_EMAIL.replace("See the attached W-2.", f"SSN {CANARY} " + "y" * 100)
    loaded = load(
        doc(f"SSN {CANARY} " + "x" * 100, name=f"{CANARY}.txt"),
        doc(email, name=f"{CANARY}.eml"),
        doc(pf.make_text_pdf([f"SSN {CANARY}"] * 5), name=f"{CANARY}.pdf"),
        ref(pf.make_text_pdf([f"SSN {CANARY}"]), name=f"{CANARY}-flat.pdf"),
        ref(filled_form(), name=f"{CANARY}-filled.pdf"),
        limits=Limits(max_source_chars=40),
    )
    assert len(loaded.warnings) >= 5
    for warning in loaded.warnings:
        assert CANARY not in warning and "987" not in warning
        assert re.match(r"S[1-9]\d*[: ]", warning)  # refers to the source by ID


def test_errors_never_contain_file_contents():
    for item in (
        data(f'{{"ssn": "{CANARY}",}}'),
        data(f"ssn,ssn\n{CANARY},1\n", name="d.csv"),
        data(f'[{{"ssn": "{CANARY}"}}, {{"ssn": "1"}}]'),
        doc(f"{CANARY}\x00".encode()),
        doc(f"{CANARY} caf\xe9".encode("latin-1")),
        ref(f"SSN {CANARY}".encode()),
        note(f"{CANARY} " * 2000),
    ):
        assert CANARY not in error(item)


def test_logs_never_contain_file_names_or_contents(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG, logger="pdf_wizard")
    load(
        doc(f"SSN {CANARY}", name=f"{CANARY}.txt"),
        data(f'{{"ssn": "{CANARY}"}}', name=f"{CANARY}.json"),
        ref(filled_form(), name=f"{CANARY}.pdf"),
        note(f"ssn {CANARY}"),
    )
    with pytest.raises(InvalidInputError):
        load(doc(b"%PDF-1.7\n" + CANARY.encode(), name=f"{CANARY}.pdf"))
    assert any(r.name.startswith("pdf_wizard.sources") for r in caplog.records)  # IDs and counts only
    assert CANARY not in caplog.text


def test_source_input_repr_hides_name_and_content():
    item = SourceInput("data", f"{CANARY}.csv", CANARY.encode(), row=2)
    assert repr(item) == "SourceInput(role='data', row=2)"
