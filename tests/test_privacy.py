"""Tests for privacy: sensitive-value detection, masking, and free-text scrubbing."""

from __future__ import annotations

import time

import pytest

from pdf_wizard.models import FieldType, FormField
from pdf_wizard.privacy import ValueRedactor, is_sensitive, luhn_valid, mask_value, scrub_text


def field(label: str | None = None, name: str = "field1", **kwargs) -> FormField:
    return FormField(id="F1", name=name, label=label, type=kwargs.pop("type", FieldType.TEXT), **kwargs)


NEUTRAL = field("City", name="city")


# --------------------------------------------------------------------------- #
# is_sensitive: value patterns
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value",
    [
        "123-45-6789",  # SSN
        "123.45.6789",  # dotted SSN
        "123/45/6789",  # slashed SSN
        "123\u200b-45-6789",  # zero-width space inside SSN
        "123456789",  # SSN without dashes
        "912-70-1234",  # ITIN
        "12-3456789",  # EIN
        "4111 1111 1111 1111",  # card (Luhn)
        "4111-1111-1111-1111",
        "4111.1111.1111.1111",
        "4111/1111/1111/1111",
        "jane.doe@example.com",
        "Account 12 34 56 78",  # 8 digits with separators
        "acct 1234.5678.9012",
        "IBAN DE89370400440532013000",
        "SSN: 123-45-6789 (on file)",  # embedded
        "DOB 01/02/1980",
        "Date of birth: January 2, 1980",
    ],
)
def test_sensitive_value_patterns(value: str) -> None:
    assert is_sensitive(NEUTRAL, value)


@pytest.mark.parametrize(
    "value",
    [
        "Jane Doe",
        "Springfield",
        "1234567",
        "12/31/1990",
        "2024-01-02",
        "01.02.2024",
        "01/02/1980 to 03/04/1990",
        "1234567.89",  # plain decimal amount
        "12345678.90",
        "Apt 4B",
        "",
        "checked",
    ],
)
def test_ordinary_values_are_not_sensitive(value: str) -> None:
    assert not is_sensitive(NEUTRAL, value)


def test_luhn() -> None:
    assert luhn_valid("4111 1111 1111 1111")
    assert luhn_valid("79927398713")
    assert not luhn_valid("4111 1111 1111 1112")
    assert not luhn_valid("")
    assert not luhn_valid("no digits")


# --------------------------------------------------------------------------- #
# is_sensitive: labels and names
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "label, name",
    [
        ("Social Security Number", "f1_01"),
        ("SSN", "f1_01"),
        (None, "f1_ssn[0]"),
        (None, "applicantDOB"),
        ("D.O.B.", "x"),
        ("Date of birth", "x"),
        ("Birth date", "x"),
        ("Born", "x"),
        ("Driver's License #", "x"),
        ("Drivers licence number", "x"),
        ("Passport No.", "x"),
        ("Tax ID", "x"),
        ("TIN", "x"),
        ("EIN", "x"),
        ("ITIN", "x"),
        ("Bank account", "x"),
        ("Acct #", "x"),
        ("Routing number", "x"),
        ("IBAN", "x"),
        ("Card number", "x"),
        ("CVV", "x"),
        ("PIN", "x"),
        ("Password", "x"),
    ],
)
def test_sensitive_labels(label: str | None, name: str) -> None:
    assert is_sensitive(field(label, name=name), "on file")


@pytest.mark.parametrize("label", ["Spinning class", "Opinion", "Destination", "Stein Street", "Mountain view", "City"])
def test_label_terms_match_whole_words_only(label: str) -> None:
    assert not is_sensitive(field(label, name="x"), "on file")


def test_checkbox_and_radio_labels_are_not_checked() -> None:
    # "checked" / "married" are not identifiers, whatever the label says.
    assert not is_sensitive(field("SSN on file?", type=FieldType.CHECKBOX), "checked")
    assert not is_sensitive(field("Account type", type=FieldType.RADIO), "savings")


def test_password_fields_and_ssn_hint_are_sensitive() -> None:
    assert is_sensitive(field("Code", password=True), "abc")
    assert is_sensitive(field("Number", format_hint="ssn"), "abc")


# --------------------------------------------------------------------------- #
# mask_value
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value, masked",
    [
        ("123-45-6789", "***-**-6789"),
        ("123.45.6789", "***.**.6789"),
        ("123/45/6789", "***/**/6789"),
        ("123\u200b-45-6789", "***-**-6789"),
        ("4111 1111 1111 1111", "**** **** **** 1111"),
        ("12345678", "****5678"),
        ("1234567", "*******"),  # fewer than 8 letters/digits: fully masked
        ("01/02/1980", "**/**/****"),  # date-shaped: fully masked (year not revealed)
        ("1980-01-02", "****-**-**"),
        ("January 2, 1980", "******* *, ****"),
        ("Jane", "****"),
        ("jane.doe@example.com", "****.***@******e.com"),
        ("José", "****"),
        ("---", "***"),
        ("***", "****"),
        ("", ""),
    ],
)
def test_mask_value(value: str, masked: str) -> None:
    assert mask_value(value) == masked


@pytest.mark.parametrize("value", ["a", "1", "-", "*", " ", "\n", "\u200b", "**", "Jose\u0301", "x" * 50])
def test_mask_value_never_returns_input_unchanged(value: str) -> None:
    assert mask_value(value) != value


# --------------------------------------------------------------------------- #
# scrub_text
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text, scrubbed",
    [
        ("My SSN is 123-45-6789.", "My SSN is ***-**-6789."),
        ("SSN 123.45.6789", "SSN ***.**.6789"),
        ("SSN 123/45/6789", "SSN ***/**/6789"),
        ("DOB 01/02/1980", "DOB **/**/****"),
        ("D.O.B.: 01/02/80", "D.O.B.: **/**/**"),
        ("Date of birth (MM/DD/YYYY): 01/02/1980", "Date of birth (MM/DD/YYYY): **/**/****"),
        ("date of birth\n1980-01-02", "date of birth\n****-**-**"),
        ("Birth date 1.2.1980", "Birth date *.*.****"),
        ("Birthdate: 02-Jan-1980", "Birthdate: **-***-****"),
        ("born on January 2, 1980", "born on ******* *, ****"),
        ("Born 2nd of January 1980", "Born *** ** ******* ****"),
        ("Born 12/31/1990 in Ohio", "Born **/**/**** in Ohio"),
        ("acct 1234.5678.9012", "acct ****.****.9012"),
        ("card 4111.1111.1111.1111", "card ****.****.****.1111"),
        ("Card 4111 1111 1111 1111 ok", "Card **** **** **** 1111 ok"),
        ("Contact jane@example.com today", "Contact ****@******e.com today"),
        ("Account 12345678 closed", "Account ****5678 closed"),
        ("EIN 12-3456789", "EIN **-***6789"),
        ("Reference: 123\u200b-45-6789", "Reference: ***-**-6789"),
        ("Order 1234567 shipped", "Order 1234567 shipped"),
        ("Amount 1234567.89 paid", "Amount 1234567.89 paid"),
        ("Signed 12/31/1990 in Ohio", "Signed 12/31/1990 in Ohio"),
        ("Date signed: January 2, 2024", "Date signed: January 2, 2024"),
        ("Range 01/02/1980 03/04/1990", "Range 01/02/1980 03/04/1990"),
        ("Place of birth: Ohio (visited 01/02/2024)", "Place of birth: Ohio (visited 01/02/2024)"),
        ("stubborn 01/02/2024", "stubborn 01/02/2024"),
        ("Nothing to see here", "Nothing to see here"),
        ("", ""),
    ],
)
def test_scrub_text(text: str, scrubbed: str) -> None:
    assert scrub_text(text) == scrubbed


def test_scrub_text_is_idempotent() -> None:
    text = (
        "SSN 123-45-6789, SSN2 123.45.6789, DOB 01/02/1980, "
        "card 4111-1111-1111-1111, mail a.b@example.org, acct 9876543210"
    )
    once = scrub_text(text)
    for secret in ("123-45-6789", "123.45.6789", "01/02/1980", "a.b@example.org", "9876543210"):
        assert secret not in once
    assert scrub_text(once) == once


def test_value_redactor_matches_variants_and_respects_boundaries() -> None:
    redactor = ValueRedactor(["X1234567", "01/02/1980", "1234", "p@ss.w0rd", "ab"])
    assert redactor("Passport x 1234567 or X-1234567") == "Passport * ***4567 or *-***4567"
    assert redactor("Born January 2, 1980 (1980-01-02)") == "Born ******* *, **** (****-**-**)"
    assert redactor("Invoice 51234 vs PIN 1234") == "Invoice 51234 vs PIN ****"
    assert redactor("Secret p@ss.w0rd here") == "Secret *@**.**** here"
    # "ab" has fewer than 3 letters/digits and is ignored.
    assert redactor("ab cab") == "ab cab"


@pytest.mark.parametrize(
    "text",
    [
        "1-" * 50_000,
        "1 " * 50_000,
        "1." * 50_000,
        "1/" * 50_000,
        "01/02/1980 " * 10_000,
        "DOB " * 20_000,
        "a" * 100_000 + "@",
        "a." * 50_000 + "@x",
    ],
)
def test_scrub_text_is_fast_on_adversarial_input(text: str) -> None:
    started = time.perf_counter()
    scrub_text(text)
    assert is_sensitive(NEUTRAL, text) in (True, False)
    assert time.perf_counter() - started < 5
