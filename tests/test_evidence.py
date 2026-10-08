"""Tests for value support: the evidence must state the proposed value."""

from __future__ import annotations

import pytest

from pdf_wizard.evidence import value_supported
from pdf_wizard.models import FieldType, FormField, Option


def text_field(**kwargs) -> FormField:
    return FormField(id="F1", name="f", type=FieldType.TEXT, **kwargs)


def supported(value: str, evidence: str, f: FormField | None = None, pdf_value: str | None = None) -> bool:
    f = f or text_field()
    return value_supported(f, value, pdf_value if pdf_value is not None else value, values=evidence, context=evidence)


@pytest.mark.parametrize(
    "value, evidence",
    [
        ("Jane Doe", "Applicant: Jane Doe"),
        ("JANE DOE", "applicant: jane doe"),
        ("Jane Doe", "Doe, Jane"),  # reordered
        ("Jane", "Jane Q. Public"),  # dropped words
        ("Jane Q Public", "Jane Quincy Public"),  # initial
        ("Jose Garcia", "José García"),  # accents
        ("Mary-Jane O'Neil", "Mary Jane O\u2019Neil"),
        ("123 Main St", "lives at 123 Main Street"),
        ("123 Main Street, Apt 4", "123 Main St. Apt. 4"),
        ("TX", "Austin, Texas 78701"),
        ("New York", "Brooklyn, NY 11201"),
        ("NY", "New York, New York"),
        ("US", "Country: United States of America"),
        ("USA", "Citizenship: U.S.A."),
        ("PO Box 12", "P.O. Box 12"),
        ("123456789", "SSN: 123-45-6789"),
        ("123", "SSN: 123456789"),  # split 3-2-4 SSN fields (e.g. IRS Form W-9)
        ("45", "SSN: 123456789"),
        ("6789", "SSN: 123456789"),
        ("98", "EIN: 987654321"),  # split 2-7 EIN fields
        ("7654321", "EIN: 987654321"),
        ("415", "Phone: 4155550199"),  # split 3-3-4 phone fields
        ("555", "Phone: 4155550199"),
        ("0199", "Phone: 4155550199"),
        ("123-45-6789", "SSN 123 45 6789"),
        ("(555) 123-4567", "Phone: +1 555.123.4567"),
        ("5551234567", "call 555-123-4567"),
        ("1234.56", "Total: $1,234.56"),
        ("25%", "a 25 % share"),
        ("78701", "Austin, TX 78701-1234"),
        ("01/02/1980", "born January 2, 1980"),
        ("1980-01-02", "DOB: 01/02/1980"),
        ("January 2, 1980", "DOB 1980-01-02"),
        ("2 Jan 1980", "dob: 01.02.1980"),
        ("3/7/24", "Date: March 7, 2024"),
        ("01021980", "Born on Jan. 2nd, 1980"),
        ("10/06/2026", "Today's date: 2026-10-06 (Tuesday, October 6, 2026)."),
        ("Tuesday, October 6, 2026", "Today's date: 2026-10-06 (Tuesday, October 6, 2026)."),
        ("03/2024", "Started in March 2024"),
        ("1980", "DOB 01/02/1980"),
        ("jane.doe@example.com", "Email: Jane.Doe@Example.com"),
        ("Yes", "Citizenship: United States"),  # yes/no answers judge the evidence
        ("王小明", "申请人王小明先生"),  # scripts without spaces
        ("?!", "anything"),  # nothing to compare
    ],
)
def test_supported(value: str, evidence: str) -> None:
    assert supported(value, evidence)


@pytest.mark.parametrize(
    "value, evidence",
    [
        ("Mallory Attacker", "Dear Jane Doe"),
        ("Jane Doe", "Jane Smith"),
        ("Jane Quincy Public", "Jane Q. Public"),  # adds information
        ("999-88-7777", "Jane Doe"),
        ("999887777", "SSN: 123-45-6789"),
        ("1", "Born 2019"),  # digits must be whole numbers, not parts of one
        ("678", "Phone 555-6789"),  # whole numbers only; "6789" alone would be fine
        ("12346", "SSN 123-45-6789"),
        ("01/03/1980", "born January 2, 1980"),
        ("01/02/1981", "DOB 01/02/1980"),
        ("payee@attacker.example", "Jane Doe"),
        ("doe.jane@example.com", "Email: jane.doe@example.com"),  # order matters in addresses
        ("jane.doe@example.org", "Email: jane.doe@example.com"),
        ("DE00 0000 0000 0000 0000 00", "Jane Doe"),
        ("Springfield", "Jane Doe ... Main"),
    ],
)
def test_not_supported(value: str, evidence: str) -> None:
    assert not supported(value, evidence)


def test_reformatted_value_may_match_through_the_pdf_value() -> None:
    # Percent fields store the fraction; the model's "25%" is what the evidence states.
    assert supported("25%", "a 25% share", pdf_value="0.25")
    # An ISO date rendered in the field's format.
    assert supported("1980-01-02", "DOB: January 2, 1980", pdf_value="Jan 2, 1980")


def test_checkboxes_are_exempt() -> None:
    f = FormField(id="F1", name="c", type=FieldType.CHECKBOX, options=[Option(value="Off"), Option(value="Yes")])
    assert supported("checked", "Marital status: married", f, pdf_value="Yes")


def radio(*options: tuple[str, str | None]) -> FormField:
    return FormField(
        id="F1", name="r", type=FieldType.RADIO, options=[Option(value=v, label=label) for v, label in options]
    )


@pytest.mark.parametrize(
    "f, pdf_value, evidence, ok",
    [
        (radio(("self", "Self-employed"), ("employed", "Employed")), "self", "self-employed since 2015", True),
        (radio(("self", "Self-employed"), ("employed", "Employed")), "self", "Employer: Acme Corp", False),
        (radio(("single", "Single"), ("married", "Married filing jointly")), "married", "Status: married", True),
        (radio(("M", None), ("F", None)), "F", "Gender: female", True),  # single letters may be initials
        (radio(("Choice1", None), ("Choice2", None)), "Choice2", "anything", True),  # no meaning of its own
        (radio(("0", None), ("1", None)), "1", "anything", True),
        (radio(("1", "Yes"), ("2", "No")), "1", "Citizenship: United States", True),  # yes/no answers
    ],
)
def test_radio_options(f: FormField, pdf_value: str, evidence: str, ok: bool) -> None:
    assert supported(pdf_value, evidence, f, pdf_value=pdf_value) is ok


def test_choice_options() -> None:
    f = FormField(
        id="F1",
        name="state",
        type=FieldType.CHOICE,
        options=[Option(value="TX", label="Texas"), Option(value="IN", label="Indiana"), Option(value="2", label="2")],
    )
    assert supported("TX", "Austin, Texas", f)
    assert supported("TX", "Austin, TX 78701", f)
    assert not supported("TX", "Columbus, Ohio", f)
    assert supported("IN", "Indianapolis, Indiana", f)  # a stopword-like code still counts
    assert supported("2", "2 dependents", f)
    assert not supported("2", "Born 2019", f)  # digits are meaningful in choice lists
    # An editable combo box value that isn't an option is checked as text.
    combo = FormField(id="F1", name="c", type=FieldType.CHOICE, editable=True, options=[Option(value="TX")])
    assert supported("Ohio", "Columbus, Ohio", combo)
    assert not supported("Ohio", "Austin, Texas", combo)


def test_structured_key_paths_can_name_an_option() -> None:
    f = radio(("self", "Self-employed"), ("employed", "Employed"))
    assert value_supported(f, "self", "self", values="true", context="employment.self_employed true")
    assert not value_supported(f, "self", "self", values="true", context="applicant.name Jane Doe")
