"""Synthetic dataset & fake filled PDF generator using Faker.

Generates 10s or 100s of realistic fake records for built-in forms (`irs_w9`,
`irs_w4`, `irs_8822`, `irs_w8ben`, `job_application`, or `all`) or any custom
AcroForm PDF (`--form path/to/form.pdf`).

For every generated record, writes:
1. A filled ground-truth PDF in `<out>/filled/<id>.pdf` (via `fill_pdf`).
2. An unstructured source document in `<out>/cases/<id>/sources/` using the
   chosen modality (`txt`, `scanned_pdf`, `jpeg`, `png`, `pdf`, `eml`).
3. A complete `evals` case (`form.pdf`, `case.json`, `expected.json`) under
   `<out>/cases/<id>/` so `python -m evals.run --cases <out>/cases` can
   benchmark models at scale.

Usage::

    .venv/bin/python -m evals.dataset --template irs_w9 --count 100 --out /tmp/w9_dataset
    .venv/bin/python -m evals.dataset --template all --count 50 --formats txt,scanned_pdf,jpeg,png --out /tmp/dataset
    .venv/bin/python -m evals.dataset --form my_form.pdf --count 100 --out /tmp/custom_dataset
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import sys
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, TextIO

from faker import Faker
from PIL import Image, ImageDraw, ImageFont
from reportlab import rl_config
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from pdf_wizard.config import Settings
from pdf_wizard.models import FieldType, FormField
from pdf_wizard.pdf.filler import fill_pdf
from pdf_wizard.pdf.inspector import LoadedForm, open_form
from pdf_wizard.schema import apply_schema, detect_schema

from .harness import CASES_DIR

rl_config.invariant = 1

TEMPLATES: dict[str, Path] = {
    "irs_w9": CASES_DIR / "irs_w9" / "form.pdf",
    "irs_w4": CASES_DIR / "irs_w4" / "form.pdf",
    "irs_8822": CASES_DIR / "irs_8822" / "form.pdf",
    "irs_w8ben": CASES_DIR / "irs_w8ben" / "form.pdf",
    "job_application": CASES_DIR / "job_application" / "form.pdf",
}

FORMATS = ("txt", "scanned_pdf", "jpeg", "png", "pdf", "eml")

_US_STATES = (
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
)

_TREATY_COUNTRIES: tuple[tuple[str, str, str, str], ...] = (
    ("France", "fr_FR", "Article 12(1)", "0"),
    ("Germany", "de_DE", "Article 12(1)", "0"),
    ("United Kingdom", "en_GB", "Article 12(1)", "0"),
    ("Canada", "en_CA", "Article XII(2)", "10"),
    ("Japan", "en_US", "Article 12(1)", "0"),
    ("Netherlands", "nl_NL", "Article 13(1)", "0"),
)


@dataclass(frozen=True)
class GeneratedSample:
    """One synthesized form-filling record."""

    sample_id: str
    template: str
    modality: str
    pdf_values: dict[str, str]  # field.id -> value passed to fill_pdf
    expected: dict[str, Any]  # field.name -> expected value(s) or None for expected.json
    title: str
    subtitle: str
    prose_paragraphs: list[str]
    structured_lines: list[str]
    sender_name: str
    sender_email: str
    today: str = "2026-10-07"


@dataclass(frozen=True)
class DatasetManifest:
    output_dir: Path
    count: int
    seed: int
    templates: tuple[str, ...]
    formats: tuple[str, ...]
    filled_pdfs: tuple[Path, ...]
    case_dirs: tuple[Path, ...]


# --------------------------------------------------------------------------- #
# Template generators
# --------------------------------------------------------------------------- #


def _clean_ssn(fake: Faker) -> tuple[str, str, str]:
    p1 = f"{fake.random_int(100, 899):03d}"
    if p1 == "666":
        p1 = "665"
    p2 = f"{fake.random_int(10, 99):02d}"
    p3 = f"{fake.random_int(1000, 9999):04d}"
    return p1, p2, p3


def _clean_ein(fake: Faker) -> tuple[str, str]:
    return f"{fake.random_int(10, 98):02d}", f"{fake.random_int(1000000, 9999999):07d}"


def _clean_phone(fake: Faker) -> str:
    return f"({fake.random_int(201, 989):03d}) {fake.random_int(200, 999):03d}-{fake.random_int(1000, 9999):04d}"


def _clean_street(fake: Faker) -> str:
    return f"{fake.building_number()} {fake.street_name()} {fake.street_suffix()}"


def _generate_irs_w9(
    sample_id: str,
    modality: str,
    form: LoadedForm,
    fake: Faker,
    rng: random.Random,
) -> GeneratedSample:
    first = fake.first_name()
    mid = rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
    last = fake.last_name()
    full_name = f"{first} {mid}. {last}"
    biz_name = f"{last} {rng.choice(('Consulting', 'Design', 'Analytics', 'Systems', 'Ventures'))} LLC"
    street = _clean_street(fake)
    city = fake.city()
    state = rng.choice(_US_STATES)
    zip_code = f"{fake.random_int(10000, 99950):05d}"
    city_state_zip = f"{city}, {state} {zip_code}"
    req_company = f"{fake.last_name()} {rng.choice(('Corporation', 'Holdings Inc.', 'Global LLC'))}"
    req_street = _clean_street(fake)
    req_city = fake.city()
    req_state = rng.choice(_US_STATES)
    req_zip = f"{fake.random_int(10000, 99950):05d}"
    requester = f"{req_company}, {req_street}, {req_city}, {req_state} {req_zip}"
    acct = f"VND-2026-{fake.random_int(1000, 9999)}"
    ssn1, ssn2, ssn3 = _clean_ssn(fake)
    ssn = f"{ssn1}-{ssn2}-{ssn3}"

    by_name = {f.name: f for f in form.info.fields}
    pdf_values = {
        by_name["topmostSubform[0].Page1[0].f1_01[0]"].id: full_name,
        by_name["topmostSubform[0].Page1[0].f1_02[0]"].id: biz_name,
        by_name["topmostSubform[0].Page1[0].Boxes3a-b_ReadOrder[0].c1_1[0]"].id: "1",
        by_name["topmostSubform[0].Page1[0].Address_ReadOrder[0].f1_07[0]"].id: street,
        by_name["topmostSubform[0].Page1[0].Address_ReadOrder[0].f1_08[0]"].id: city_state_zip,
        by_name["topmostSubform[0].Page1[0].f1_09[0]"].id: requester,
        by_name["topmostSubform[0].Page1[0].f1_10[0]"].id: acct,
        by_name["topmostSubform[0].Page1[0].f1_11[0]"].id: ssn1,
        by_name["topmostSubform[0].Page1[0].f1_12[0]"].id: ssn2,
        by_name["topmostSubform[0].Page1[0].f1_13[0]"].id: ssn3,
    }
    expected: dict[str, Any] = {
        "topmostSubform[0].Page1[0].f1_01[0]": [full_name, f"{first} {mid} {last}"],
        "topmostSubform[0].Page1[0].f1_02[0]": biz_name,
        "topmostSubform[0].Page1[0].Boxes3a-b_ReadOrder[0].c1_1[0]": True,
        "topmostSubform[0].Page1[0].Boxes3a-b_ReadOrder[0].c1_1[1]": None,
        "topmostSubform[0].Page1[0].Boxes3a-b_ReadOrder[0].c1_1[2]": None,
        "topmostSubform[0].Page1[0].Address_ReadOrder[0].f1_07[0]": street,
        "topmostSubform[0].Page1[0].Address_ReadOrder[0].f1_08[0]": [city_state_zip, f"{city} {state} {zip_code}"],
        "topmostSubform[0].Page1[0].f1_09[0]": [
            requester,
            f"{req_company}\n{req_street}, {req_city}, {req_state} {req_zip}",
            f"{req_company}\n{req_street}\n{req_city}, {req_state} {req_zip}",
        ],
        "topmostSubform[0].Page1[0].f1_10[0]": acct,
        "topmostSubform[0].Page1[0].f1_11[0]": ssn1,
        "topmostSubform[0].Page1[0].f1_12[0]": ssn2,
        "topmostSubform[0].Page1[0].f1_13[0]": ssn3,
        "topmostSubform[0].Page1[0].f1_14[0]": None,
        "topmostSubform[0].Page1[0].f1_15[0]": None,
    }
    prose = [
        f"Hi Accounts Payable, I am submitting my W-9 details for vendor onboarding. My legal name is {full_name}, and my business / disregarded entity name is {biz_name} (Federal tax classification: Individual/sole proprietor).",
        f"My street address is {street}, and my city, state, and ZIP code is {city_state_zip}. The requester's name and address is {requester}, and my vendor account number is {acct}. My Social Security Number (SSN) is {ssn}.",
    ]
    lines = [
        f"Legal Name of Individual / Owner (Line 1): {full_name}",
        f"Business / Disregarded Entity Name (Line 2): {biz_name}",
        "Federal Tax Classification (Line 3a): Individual/sole proprietor",
        f"Street Address (Line 5): {street}",
        f"City, State, and ZIP Code (Line 6): {city_state_zip}",
        f"Requester Name and Address: {requester}",
        f"List Account Number(s) (Line 7): {acct}",
        f"Social Security Number (SSN): {ssn}",
    ]
    return GeneratedSample(
        sample_id=sample_id,
        template="irs_w9",
        modality=modality,
        pdf_values=pdf_values,
        expected=expected,
        title="VENDOR TAX INFORMATION SHEET (FORM W-9)",
        subtitle=f"Contractor Onboarding Record — {sample_id}",
        prose_paragraphs=prose,
        structured_lines=lines,
        sender_name=full_name,
        sender_email=f"{first.lower()}.{last.lower()}@example.com",
    )


def _generate_irs_w4(
    sample_id: str,
    modality: str,
    form: LoadedForm,
    fake: Faker,
    rng: random.Random,
) -> GeneratedSample:
    first = fake.first_name()
    mid = rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
    first_mid = f"{first} {mid}."
    last = fake.last_name()
    street = _clean_street(fake)
    city = fake.city()
    state = rng.choice(_US_STATES)
    zip_code = f"{fake.random_int(10000, 99950):05d}"
    city_state_zip = f"{city}, {state} {zip_code}"
    ssn1, ssn2, ssn3 = _clean_ssn(fake)
    ssn = f"{ssn1}-{ssn2}-{ssn3}"

    status_idx, status_export, status_label = rng.choice((
        (0, "1", "Single or Married filing separately"),
        (1, "2", "Married filing jointly"),
        (2, "3", "Head of household"),
    ))
    num_kids = rng.choice((1, 2, 3))
    kids_amt = str(num_kids * 2200)
    other_dep_amt = str(rng.choice((0, 500, 1000)))
    total_credits = str(int(kids_amt) + int(other_dep_amt))
    other_inc = str(rng.choice((500, 1200, 2500)))
    deductions = "0"
    extra_withholding = str(rng.choice((25, 50, 75, 100)))

    emp_company = f"{fake.last_name()} {rng.choice(('Cloud Systems Inc.', 'Technologies Corp.', 'Robotics LLC'))}"
    emp_street = _clean_street(fake)
    emp_addr = f"{emp_company}, {emp_street}, {city}, {state} {zip_code}"
    start_date = f"{fake.random_int(1, 12):02d}/{fake.random_int(1, 28):02d}/2026"
    ein1, ein2 = _clean_ein(fake)
    ein = f"{ein1}-{ein2}"

    by_name = {f.name: f for f in form.info.fields}
    pdf_values = {
        by_name["topmostSubform[0].Page1[0].Step1a[0].f1_01[0]"].id: first_mid,
        by_name["topmostSubform[0].Page1[0].Step1a[0].f1_02[0]"].id: last,
        by_name["topmostSubform[0].Page1[0].Step1a[0].f1_03[0]"].id: street,
        by_name["topmostSubform[0].Page1[0].Step1a[0].f1_04[0]"].id: city_state_zip,
        by_name["topmostSubform[0].Page1[0].f1_05[0]"].id: ssn,
        by_name[f"topmostSubform[0].Page1[0].c1_1[{status_idx}]"].id: status_export,
        by_name["topmostSubform[0].Page1[0].c1_2[0]"].id: "1",
        by_name["topmostSubform[0].Page1[0].Step3_ReadOrder[0].f1_06[0]"].id: kids_amt,
        by_name["topmostSubform[0].Page1[0].Step3_ReadOrder[0].f1_07[0]"].id: other_dep_amt,
        by_name["topmostSubform[0].Page1[0].f1_08[0]"].id: total_credits,
        by_name["topmostSubform[0].Page1[0].f1_09[0]"].id: other_inc,
        by_name["topmostSubform[0].Page1[0].f1_10[0]"].id: deductions,
        by_name["topmostSubform[0].Page1[0].f1_11[0]"].id: extra_withholding,
        by_name["topmostSubform[0].Page1[0].f1_12[0]"].id: emp_addr,
        by_name["topmostSubform[0].Page1[0].f1_13[0]"].id: start_date,
        by_name["topmostSubform[0].Page1[0].f1_14[0]"].id: ein,
    }
    expected: dict[str, Any] = {
        "topmostSubform[0].Page1[0].Step1a[0].f1_01[0]": [first_mid, f"{first} {mid}"],
        "topmostSubform[0].Page1[0].Step1a[0].f1_02[0]": last,
        "topmostSubform[0].Page1[0].Step1a[0].f1_03[0]": street,
        "topmostSubform[0].Page1[0].Step1a[0].f1_04[0]": [city_state_zip, f"{city} {state} {zip_code}"],
        "topmostSubform[0].Page1[0].f1_05[0]": [ssn, f"{ssn1}{ssn2}{ssn3}"],
        "topmostSubform[0].Page1[0].c1_1[0]": True if status_idx == 0 else None,
        "topmostSubform[0].Page1[0].c1_1[1]": True if status_idx == 1 else None,
        "topmostSubform[0].Page1[0].c1_1[2]": True if status_idx == 2 else None,
        "topmostSubform[0].Page1[0].c1_2[0]": True,
        "topmostSubform[0].Page1[0].c1_3[0]": None,
        "topmostSubform[0].Page1[0].Step3_ReadOrder[0].f1_06[0]": [kids_amt, f"{int(kids_amt):,}"],
        "topmostSubform[0].Page1[0].Step3_ReadOrder[0].f1_07[0]": [other_dep_amt, f"{int(other_dep_amt):,}"],
        "topmostSubform[0].Page1[0].f1_08[0]": [total_credits, f"{int(total_credits):,}"],
        "topmostSubform[0].Page1[0].f1_09[0]": [other_inc, f"{int(other_inc):,}"],
        "topmostSubform[0].Page1[0].f1_10[0]": ["0", "0.00"],
        "topmostSubform[0].Page1[0].f1_11[0]": [extra_withholding, f"{extra_withholding}.00"],
        "topmostSubform[0].Page1[0].f1_12[0]": [
            emp_addr,
            f"{emp_company}\n{emp_street}, {city}, {state} {zip_code}",
            f"{emp_company}\n{emp_street}\n{city}, {state} {zip_code}",
        ],
        "topmostSubform[0].Page1[0].f1_13[0]": start_date,
        "topmostSubform[0].Page1[0].f1_14[0]": ein,
    }
    prose = [
        f"New hire {first_mid} {last} (SSN {ssn}) resides at {street} in {city_state_zip}. Filing status (Step 1c): {status_label}. Check Step 2(c) for two jobs total.",
        f"Step 3 Dependents: qualifying children amount is {kids_amt}, other dependents amount is {other_dep_amt}, and total dependent credits on line 3 is {total_credits}. Step 4 Other Adjustments: 4(a) other income is {other_inc}, 4(b) deductions is {deductions}, and 4(c) extra withholding is {extra_withholding}.",
        f"Employers Only section: Employer name and address is {emp_addr}, first date of employment is {start_date}, and Employer Identification Number (EIN) is {ein}.",
    ]
    lines = [
        f"First Name and Middle Initial (Step 1a): {first_mid}",
        f"Last Name (Step 1a): {last}",
        f"Street Address (Step 1a): {street}",
        f"City or Town, State, and ZIP Code (Step 1a): {city_state_zip}",
        f"Social Security Number (Step 1b): {ssn}",
        f"Filing Status (Step 1c): {status_label}",
        "Multiple Jobs (Step 2c): Check box (two jobs total)",
        f"Qualifying Children Credit Amount (Step 3a): {kids_amt}",
        f"Other Dependents Credit Amount (Step 3b): {other_dep_amt}",
        f"Total Dependent Credits (Step 3 Total): {total_credits}",
        f"Other Income (Step 4a): {other_inc}",
        f"Deductions (Step 4b): {deductions}",
        f"Extra Withholding per Pay Period (Step 4c): {extra_withholding}",
        f"Employer Name and Address: {emp_addr}",
        f"First Date of Employment: {start_date}",
        f"Employer Identification Number (EIN): {ein}",
    ]
    return GeneratedSample(
        sample_id=sample_id,
        template="irs_w4",
        modality=modality,
        pdf_values=pdf_values,
        expected=expected,
        title="EMPLOYEE WITHHOLDING SUMMARY (FORM W-4)",
        subtitle=f"Payroll Onboarding Record — {sample_id}",
        prose_paragraphs=prose,
        structured_lines=lines,
        sender_name=f"{first_mid} {last}",
        sender_email=f"{first.lower()}.{last.lower()}@example.com",
    )


def _generate_irs_8822(
    sample_id: str,
    modality: str,
    form: LoadedForm,
    fake: Faker,
    rng: random.Random,
) -> GeneratedSample:
    last = fake.last_name()
    p1_first = fake.first_name()
    p1_mid = rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
    p1_name = f"{p1_first} {p1_mid}. {last}"
    p1_ssn = "-".join(_clean_ssn(fake))

    p2_first = fake.first_name()
    p2_mid = rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
    p2_name = f"{p2_first} {p2_mid}. {last}"
    p2_ssn = "-".join(_clean_ssn(fake))

    old_addr = f"{_clean_street(fake)}, {fake.city()}, {rng.choice(_US_STATES)} {fake.random_int(10000, 99950):05d}"
    new_addr = f"{_clean_street(fake)}, {fake.city()}, {rng.choice(_US_STATES)} {fake.random_int(10000, 99950):05d}"
    phone = _clean_phone(fake)
    phone_dash = phone.replace("(", "").replace(") ", "-")

    by_name = {f.name: f for f in form.info.fields}
    pdf_values = {
        by_name["topmostSubform[0].Page1[0].c1_1[0]"].id: "1",
        by_name["topmostSubform[0].Page1[0].f1_3[0]"].id: p1_name,
        by_name["topmostSubform[0].Page1[0].f1_4[0]"].id: p1_ssn,
        by_name["topmostSubform[0].Page1[0].f1_5[0]"].id: p2_name,
        by_name["topmostSubform[0].Page1[0].f1_6[0]"].id: p2_ssn,
        by_name["topmostSubform[0].Page1[0].f1_9[0]"].id: old_addr,
        by_name["topmostSubform[0].Page1[0].f1_17[0]"].id: new_addr,
        by_name["topmostSubform[0].Page1[0].f1_21[0]"].id: phone,
    }
    expected: dict[str, Any] = {
        "topmostSubform[0].Page1[0].c1_1[0]": True,
        "topmostSubform[0].Page1[0].c1_2[0]": None,
        "topmostSubform[0].Page1[0].c1_3[0]": None,
        "topmostSubform[0].Page1[0].f1_3[0]": p1_name,
        "topmostSubform[0].Page1[0].f1_4[0]": p1_ssn,
        "topmostSubform[0].Page1[0].f1_5[0]": p2_name,
        "topmostSubform[0].Page1[0].f1_6[0]": p2_ssn,
        "topmostSubform[0].Page1[0].f1_7[0]": None,
        "topmostSubform[0].Page1[0].f1_9[0]": old_addr,
        "topmostSubform[0].Page1[0].f1_17[0]": new_addr,
        "topmostSubform[0].Page1[0].f1_21[0]": [phone, phone_dash],
    }
    prose = [
        "Please file IRS Form 8822 to change our home mailing address for our individual income tax returns (Form 1040, check line 1).",
        f"Primary taxpayer (Line 3a): {p1_name} (SSN {p1_ssn}). Spouse (Line 4a): {p2_name} (SSN {p2_ssn}).",
        f"Our old home mailing address (Line 6a) was {old_addr}, and our new home mailing address (Line 7) is {new_addr}. Daytime telephone number: {phone}.",
    ]
    lines = [
        "Return Type (Line 1): Individual income tax returns (Form 1040)",
        f"Primary Taxpayer Name (Line 3a): {p1_name}",
        f"Primary Taxpayer SSN (Line 3b): {p1_ssn}",
        f"Spouse Full Name (Line 4a): {p2_name}",
        f"Spouse SSN (Line 4b): {p2_ssn}",
        f"Old Mailing Address (Line 6a): {old_addr}",
        f"New Mailing Address (Line 7): {new_addr}",
        f"Daytime Telephone Number: {phone}",
    ]
    return GeneratedSample(
        sample_id=sample_id,
        template="irs_8822",
        modality=modality,
        pdf_values=pdf_values,
        expected=expected,
        title="CHANGE OF HOME MAILING ADDRESS (FORM 8822)",
        subtitle=f"Taxpayer Address Update — {sample_id}",
        prose_paragraphs=prose,
        structured_lines=lines,
        sender_name=p1_name,
        sender_email=f"{p1_first.lower()}.{last.lower()}@example.com",
    )


def _generate_irs_w8ben(
    sample_id: str,
    modality: str,
    form: LoadedForm,
    fake: Faker,
    rng: random.Random,
) -> GeneratedSample:
    country, _locale, article, rate = rng.choice(_TREATY_COUNTRIES)
    full_name = f"{fake.first_name()} {fake.last_name()}"
    street = _clean_street(fake)
    postal = f"{fake.random_int(10000, 99999)}"
    city = fake.city()
    city_postal = f"{postal} {city}"
    ftin = f"{fake.random_int(1000000000000, 9999999999999)}"
    ref_num = f"INTL-{country[:2].upper()}-2026-{fake.random_int(10, 99)}"
    dob_m = f"{fake.random_int(1, 12):02d}"
    dob_d = f"{fake.random_int(1, 28):02d}"
    dob_y = f"{fake.random_int(1965, 2001)}"
    dob_dash = f"{dob_m}-{dob_d}-{dob_y}"
    dob_slash = f"{dob_m}/{dob_d}/{dob_y}"
    income_type = rng.choice(("Royalties", "Copyright royalties", "Consulting services"))
    explanation = f"Beneficial owner is a resident of {country} with no permanent establishment in the United States."

    by_name = {f.name: f for f in form.info.fields}
    pdf_values = {
        by_name["topmostSubform[0].Page1[0].f_1[0]"].id: full_name,
        by_name["topmostSubform[0].Page1[0].f_2[0]"].id: country,
        by_name["topmostSubform[0].Page1[0].f_3[0]"].id: street,
        by_name["topmostSubform[0].Page1[0].f_4[0]"].id: city_postal,
        by_name["topmostSubform[0].Page1[0].f_5[0]"].id: country,
        by_name["topmostSubform[0].Page1[0].f_10[0]"].id: ftin,
        by_name["topmostSubform[0].Page1[0].f_11[0]"].id: ref_num,
        by_name["topmostSubform[0].Page1[0].f_12[0]"].id: dob_dash,
        by_name["topmostSubform[0].Page1[0].f_13[0]"].id: country,
        by_name["topmostSubform[0].Page1[0].f_14[0]"].id: article,
        by_name["topmostSubform[0].Page1[0].f_15[0]"].id: rate,
        by_name["topmostSubform[0].Page1[0].f_16[0]"].id: income_type,
        by_name["topmostSubform[0].Page1[0].f_17[0]"].id: explanation,
        by_name["topmostSubform[0].Page1[0].c1_02[0]"].id: "1",
        by_name["topmostSubform[0].Page1[0].Date[0]"].id: "10-07-2026",
        by_name["topmostSubform[0].Page1[0].f_21[0]"].id: full_name,
    }
    expected: dict[str, Any] = {
        "topmostSubform[0].Page1[0].f_1[0]": full_name,
        "topmostSubform[0].Page1[0].f_2[0]": country,
        "topmostSubform[0].Page1[0].f_3[0]": street,
        "topmostSubform[0].Page1[0].f_4[0]": [city_postal, f"{city} {postal}", f"{city}, {postal}"],
        "topmostSubform[0].Page1[0].f_5[0]": country,
        "topmostSubform[0].Page1[0].f_6[0]": None,
        "topmostSubform[0].Page1[0].f_7[0]": None,
        "topmostSubform[0].Page1[0].f_8[0]": None,
        "topmostSubform[0].Page1[0].f_9[0]": None,
        "topmostSubform[0].Page1[0].c1_01[0]": None,
        "topmostSubform[0].Page1[0].f_10[0]": ftin,
        "topmostSubform[0].Page1[0].f_11[0]": ref_num,
        "topmostSubform[0].Page1[0].f_12[0]": [dob_dash, dob_slash],
        "topmostSubform[0].Page1[0].f_13[0]": country,
        "topmostSubform[0].Page1[0].f_14[0]": [article, article.replace("Article ", "")],
        "topmostSubform[0].Page1[0].f_15[0]": [rate, f"{rate}.0"],
        "topmostSubform[0].Page1[0].f_16[0]": [income_type, f"{income_type} income"],
        "topmostSubform[0].Page1[0].f_17[0]": explanation,
        "topmostSubform[0].Page1[0].c1_02[0]": True,
        "topmostSubform[0].Page1[0].Date[0]": ["10-07-2026", "10/07/2026"],
        "topmostSubform[0].Page1[0].f_21[0]": full_name,
    }
    prose = [
        f"Beneficial owner {full_name} (born {dob_dash}) is a citizen and resident of {country}. Permanent residence street address: {street}, city/postal code: {city_postal}, country: {country}.",
        f"Foreign tax identifying number (FTIN, Line 6a): {ftin}. Reference number (Line 7): {ref_num}.",
        f"Tax treaty claim (Lines 9 and 10): Resident of {country} claiming the provisions of {article} for a {rate}% rate of withholding on {income_type}. Additional conditions: {explanation}",
        f"Certification: I certify that I have the capacity to sign for the person identified on line 1. Print name of signer: {full_name}.",
    ]
    lines = [
        f"Name of Individual Who Is the Beneficial Owner (Line 1): {full_name}",
        f"Country of Citizenship (Line 2): {country}",
        f"Permanent Residence Street Address (Line 3): {street}",
        f"City or Town, Postal Code (Line 3): {city_postal}",
        f"Permanent Residence Country (Line 3): {country}",
        f"Foreign Tax Identifying Number / FTIN (Line 6a): {ftin}",
        f"Reference Number(s) (Line 7): {ref_num}",
        f"Date of Birth (MM-DD-YYYY) (Line 8): {dob_dash}",
        f"Tax Treaty Claim (Lines 9 and 10): Resident of {country} claiming {article} for a {rate}% rate of withholding on {income_type}.",
        f"Additional Conditions (Line 10): {explanation}",
        "Certification: I certify that I have the capacity to sign for the person identified on line 1.",
        f"Print Name of Signer: {full_name}",
    ]
    return GeneratedSample(
        sample_id=sample_id,
        template="irs_w8ben",
        modality=modality,
        pdf_values=pdf_values,
        expected=expected,
        title="FOREIGN BENEFICIAL OWNER DECLARATION (W-8BEN)",
        subtitle=f"International Onboarding Record — {sample_id}",
        prose_paragraphs=prose,
        structured_lines=lines,
        sender_name=full_name,
        sender_email=f"onboarding.{sample_id.lower()}@example.com",
    )


def _generate_job_application(
    sample_id: str,
    modality: str,
    form: LoadedForm,
    fake: Faker,
    rng: random.Random,
) -> GeneratedSample:
    first = fake.first_name()
    mid = rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
    last = fake.last_name()
    dob = f"{fake.random_int(1, 12):02d}/{fake.random_int(1, 28):02d}/{fake.random_int(1975, 2000)}"
    ssn1, ssn2, ssn3 = _clean_ssn(fake)
    ssn_dashed = f"{ssn1}-{ssn2}-{ssn3}"
    ssn_plain = f"{ssn1}{ssn2}{ssn3}"
    email = f"{first.lower()}.{last.lower()}@example.com"
    phone = _clean_phone(fake)
    phone_dash = phone.replace("(", "").replace(") ", "-")
    street = _clean_street(fake)
    city = fake.city()
    state = rng.choice(("CA", "NY", "TX", "WA"))
    zip_code = f"{fake.random_int(10000, 99950):05d}"
    role = rng.choice(("Senior Backend Engineer", "Staff Data Engineer", "Product Designer"))
    start_date = "11/02/2026"
    salary = f"${rng.choice((145, 160, 175, 190))},000"
    prev_employer = f"{fake.last_name()} {rng.choice(('Logistics', 'Systems', 'Solutions', 'Industries'))}"
    em_name = f"{fake.first_name()} {last}"
    em_phone = _clean_phone(fake)
    em_phone_dash = em_phone.replace("(", "").replace(") ", "-")

    by_name = {f.name: f for f in form.info.fields}
    pdf_values = {
        by_name["first_name"].id: first,
        by_name["middle_initial"].id: mid,
        by_name["last_name"].id: last,
        by_name["dob"].id: dob,
        by_name["ssn"].id: ssn_plain,
        by_name["email"].id: email,
        by_name["phone"].id: phone,
        by_name["street"].id: street,
        by_name["city"].id: city,
        by_name["state"].id: state,
        by_name["zip"].id: zip_code,
        by_name["position"].id: role,
        by_name["start_date"].id: start_date,
        by_name["salary"].id: salary,
        by_name["full_time"].id: "Yes",
        by_name["relocate"].id: "Yes",
        by_name["referral"].id: "friend",
        by_name["prev_employer"].id: prev_employer,
        by_name["emergency_name"].id: em_name,
        by_name["emergency_phone"].id: em_phone,
        by_name["application_date"].id: "10/07/2026",
    }
    expected: dict[str, Any] = {
        "first_name": first,
        "middle_initial": mid,
        "last_name": last,
        "dob": dob,
        "ssn": ssn_plain,
        "email": email,
        "phone": [phone, phone_dash],
        "street": street,
        "city": city,
        "state": state,
        "zip": zip_code,
        "position": role,
        "start_date": start_date,
        "salary": [salary, salary.replace("$", ""), salary.replace("$", "").replace(",", "")],
        "full_time": True,
        "relocate": True,
        "referral": "friend",
        "prev_employer": prev_employer,
        "felony": None,
        "emergency_name": em_name,
        "emergency_phone": [em_phone, em_phone_dash],
        "application_date": "10/07/2026",
    }
    prose = [
        f"Applicant {first} {mid} {last} (born {dob}, SSN {ssn_dashed}) is applying for {role} (Full-time) starting {start_date} with desired salary {salary}.",
        f"Contact: email {email}, phone {phone}, address {street}, {city}, {state} {zip_code}. Most recent employer: {prev_employer}. Referred by a friend.",
        f"Willing to relocate: Yes. Prior felony conviction: No. Emergency contact: {em_name} at {em_phone}. Application date: 10/07/2026.",
    ]
    lines = [
        f"First Name: {first}",
        f"Middle Initial: {mid}",
        f"Last Name: {last}",
        f"Date of Birth: {dob}",
        f"Social Security Number (SSN): {ssn_dashed}",
        f"Email Address: {email}",
        f"Phone Number: {phone}",
        f"Street Address: {street}",
        f"City: {city}",
        f"State: {state}",
        f"ZIP Code: {zip_code}",
        f"Position Applied For: {role}",
        f"Available Start Date: {start_date}",
        f"Desired Salary: {salary}",
        "Full-Time Position: Yes",
        "Willing to Relocate: Yes",
        "How Did You Hear About Us: Friend / referral",
        f"Most Recent Employer: {prev_employer}",
        f"Emergency Contact Name: {em_name}",
        f"Emergency Contact Phone: {em_phone}",
        "Application Date: 10/07/2026",
    ]
    return GeneratedSample(
        sample_id=sample_id,
        template="job_application",
        modality=modality,
        pdf_values=pdf_values,
        expected=expected,
        title="CANDIDATE APPLICATION SUMMARY",
        subtitle=f"Recruiting Intake Sheet — {sample_id}",
        prose_paragraphs=prose,
        structured_lines=lines,
        sender_name=f"{first} {last}",
        sender_email=email,
    )


_OVERFLOW_OR_SECONDARY_RE = re.compile(
    r"(?:\(continued\)|\bline\s+[2-9]\s+of\s+\d+\b|\(#(?:[2-9]|\d{2,})\)|\bbodyrow[2-9]\b|\brow:\s*[2-9]\b)",
    re.I,
)
_OVERFLOW_NAME_RE = re.compile(r"^(?P<base>.+?)[_.\s]*[2-9]$")


def _checkbox_stem(label: str | None) -> str:
    if not label:
        return ""
    cleaned = re.sub(r"\s*\(#\d+\)\s*$", "", label).strip(" .:")
    return cleaned.lower()


def _generate_generic(
    sample_id: str,
    template_name: str,
    modality: str,
    form: LoadedForm,
    fake: Faker,
    rng: random.Random,
) -> GeneratedSample:
    """Synthesizes realistic Faker values for any arbitrary AcroForm PDF based on its detected schema."""
    schema = detect_schema(form, enrich=False, use_cache=True)
    info = apply_schema(form.info, schema)
    pdf_values: dict[str, str] = {}
    expected: dict[str, Any] = {}
    lines: list[str] = []

    writable = [f for f in info.fields if f.writable]
    leaf_names_on_page: set[tuple[int | None, str]] = {
        (f.page, re.sub(r"\[\d+\]$", "", f.name.split(".")[-1]).lower()) for f in writable
    }

    # 1. Decide which checkbox is selected in each mutex group (prefer the first option, or "Yes" in Yes/No).
    chosen_in_group: dict[str, str] = {}
    unchecked_stems: set[str] = set()
    unchecked_yes_prefixes: set[str] = set()
    for gid, fids in schema.mutex_groups.items():
        group_fields = [f for f in writable if f.id in fids]
        if not group_fields:
            continue
        picked_cb = group_fields[0]
        chosen_in_group[gid] = picked_cb.id
        for gf in group_fields:
            if gf.id != picked_cb.id:
                stem = _checkbox_stem(gf.label)
                if len(stem) >= 4:
                    unchecked_stems.add(stem)
                if stem.endswith(" yes") and (m := re.match(r"^(\d+[a-z]?)\b", stem)):
                    unchecked_yes_prefixes.add(m.group(1))

    # Track whether a "Method of service" radio button is present (and select personal service "served").
    has_service_method_radio = any(
        f.type is FieldType.RADIO and "method" in f"{f.name} {f.label or ''}".lower() for f in writable
    )
    checked_independent_cb = False

    for f in writable:
        raw_hint = f"{f.label or ''} {f.name}".lower()
        leaf = re.sub(r"\[\d+\]$", "", f.name.split(".")[-1])

        # Skip office-use and signature fields.
        if f.office_use_only or "signature" in raw_hint or leaf.lower().startswith("sig_"):
            expected[f.name] = None
            continue

        # Skip secondary/overflow lines (e.g. "(continued)", "Line 2 of 2", "(#2)", "Row: 2", "*_2").
        if _OVERFLOW_OR_SECONDARY_RE.search(f"{f.label or ''} {f.section or ''}"):
            expected[f.name] = None
            continue
        if (m_ov := _OVERFLOW_NAME_RE.match(leaf.lower())) and (
            (f.page, m_ov.group("base")) in leaf_names_on_page
            or (f.page, f"{m_ov.group('base')}_1") in leaf_names_on_page
            or (f.page, f"{m_ov.group('base')}1") in leaf_names_on_page
            or (f.page, f"{m_ov.group('base')}.1") in leaf_names_on_page
        ):
            expected[f.name] = None
            continue

        # Skip optional "previous address", "trade name (if different)", "foreign country", "fax" distractions.
        if any(
            kw in raw_hint
            for kw in (
                "if different",
                "previous address",
                "foreign country",
                "group exemption number",
                "fax",
                "executor, administrator",
            )
        ):
            expected[f.name] = None
            continue

        # When the form has a "Method of service" radio button (personal service "served" selected),
        # leave non-personal-service branches ("left with", "left at residence", "returned unexecuted",
        # "designated by law", "organization", "other") blank.
        if has_service_method_radio and f.type is FieldType.TEXT:
            if any(
                kw in raw_hint
                for kw in (
                    "left with",
                    "left at residence",
                    "returned unexecuted",
                    "unexecuted",
                    "returned date",
                    "designated",
                    "designee",
                    "organization",
                    "served on",
                    "date served 1",
                    "date served 2",
                    "other",
                )
            ) and "place served" not in raw_hint:
                expected[f.name] = None
                continue

        # Mutex checkbox group handling.
        if f.group:
            if chosen_in_group.get(f.group) != f.id:
                expected[f.name] = None
                continue
        elif f.type is FieldType.CHECKBOX:
            # Check at most one independent checkbox per form (and skip "revoke"/"do not"/"additional" boxes).
            if checked_independent_cb or any(
                kw in raw_hint for kw in ("revoke", "retention", "specific use", "additional", "check if new")
            ):
                expected[f.name] = None
                continue
            checked_independent_cb = True

        # Skip text fields that are companion specifications for an unchecked checkbox in a mutex group.
        if f.type is FieldType.TEXT and unchecked_stems:
            f_stem = _checkbox_stem(f.label)
            if any(f_stem == s or f_stem.startswith(s + ".") or f_stem.startswith(s + " ") for s in unchecked_stems):
                expected[f.name] = None
                continue
            if any(f"if {p} is" in f_stem for p in unchecked_yes_prefixes):
                expected[f.name] = None
                continue

        val, human = _fake_value_for_field(f, fake, rng)
        if val is None:
            expected[f.name] = None
            continue
        pdf_values[f.id] = val
        if f.type is FieldType.CHECKBOX:
            expected[f.name] = True
        elif val != val.strip() and val.strip():
            expected[f.name] = [val, val.strip()]
        else:
            expected[f.name] = val
        label = f.label or f.name
        lines.append(f"{label}: {human}")

    prose = [" ".join(lines[i : i + 4]) for i in range(0, len(lines), 4)] or ["No fillable fields."]
    person = fake.name()
    return GeneratedSample(
        sample_id=sample_id,
        template=template_name,
        modality=modality,
        pdf_values=pdf_values,
        expected=expected,
        title=f"INTAKE SUMMARY — {template_name.upper()}",
        subtitle=f"Synthetic Record — {sample_id}",
        prose_paragraphs=prose,
        structured_lines=lines,
        sender_name=person,
        sender_email=fake.safe_email(),
    )


def _fake_value_for_field(f: FormField, fake: Faker, rng: random.Random) -> tuple[str | None, str]:
    hint = f"{f.label or ''} {f.name} {f.format_hint or ''}".lower()
    name_only = f.name.lower()
    if f.type is FieldType.CHECKBOX:
        on = f.on_states[0] if f.on_states else "Yes"
        return on, "Yes (checked)"
    if f.type in (FieldType.RADIO, FieldType.CHOICE):
        opts = [
            o
            for o in f.options
            if o.value and o.value.strip() and o.value.strip().lower() not in ("off", "none", "select", "choose")
        ]
        if not opts:
            return None, ""
        # For "Method of service" radio groups, prefer the first option ("served" / personal service).
        if f.type is FieldType.RADIO and "method" in hint:
            picked = opts[0]
        else:
            picked = rng.choice(opts)
        val_clean = picked.value.strip()
        lbl_clean = (picked.label or "").strip()
        if lbl_clean and val_clean.lower() not in lbl_clean.lower():
            human = f"{lbl_clean} ({val_clean})"
        else:
            human = lbl_clean or val_clean
        return picked.value, human
    # Text field
    if f.comb and f.max_len:
        digits = "".join(str(rng.randint(1, 9)) for _ in range(f.max_len))
        return digits, digits
    if f.format_hint and (m := re.match(r"^(?:number|percent):(\d+):", f.format_hint)):
        decimals = int(m.group(1))
        whole = fake.random_int(15, 250)
        num = f"{whole}.{0:0{decimals}d}" if decimals > 0 else str(whole)
        return num, num
    if f.format_hint and f.format_hint.startswith("date:") and ("h:mm" in f.format_hint.lower()):
        dt_time = f"{fake.random_int(1, 12):02d}/{fake.random_int(1, 28):02d}/2026 9:30 am"
        return dt_time, dt_time
    if f.max_len == 2 and "month" in hint:
        mo = f"{fake.random_int(1, 12):02d}"
        return mo, mo
    if f.max_len == 2 and "day" in hint:
        dy = f"{fake.random_int(1, 28):02d}"
        return dy, dy
    if f.max_len == 4 and "year" in hint:
        yr = str(fake.random_int(2021, 2025))
        return yr, yr
    if "closing month" in hint:
        return "December", "December"
    if re.search(r"\bptin\b", hint):
        ptin = f"P{fake.random_int(10000000, 99999999)}"
        return ptin, ptin
    if "caf no" in hint:
        caf = f"{fake.random_int(1000, 9999)}-{fake.random_int(10000, 99999)}R"
        return caf, caf
    if "plan number" in hint:
        return "001", "001"
    if "customer file number" in hint or (f.max_len and "numeric" in hint):
        num_str = "".join(str(rng.randint(1, 9)) for _ in range(min(f.max_len or 8, 8)))
        return num_str, num_str
    if re.search(r"\b(?:ssn|tin|itin)\b", hint) or "social security" in hint or "taxpayer identification" in hint:
        ssn = "-".join(_clean_ssn(fake))
        val = ssn.replace("-", "") if (f.max_len and f.max_len < 11) else ssn
        return val, ssn
    if re.search(r"\bein\b", hint) or "employer identification" in hint:
        ein = "-".join(_clean_ein(fake))
        val = ein.replace("-", "") if (f.max_len and f.max_len < 10) else ein
        return val, ein
    if "date" in hint or "dob" in hint or "birth" in hint:
        dt = f"{fake.random_int(1, 12):02d}/{fake.random_int(1, 28):02d}/{fake.random_int(2020, 2025)}"
        return dt, dt
    if "year(s) or period(s)" in hint:
        return "2024", "2024"
    if "tax form number" in hint or "transcript requested" in hint or "form number to be filed" in hint:
        return "1040", "1040"
    if "type of tax" in hint:
        return "Income", "Income"
    if "specific tax matters" in hint:
        return "Form 1040 Return", "Form 1040 Return"
    if any(k in hint for k in ("number of employees", "number of llc members", "agricultural", "household")) or (
        f.label and f.label.strip().startswith("13.")
    ):
        cnt = str(fake.random_int(1, 12))
        return cnt, cnt
    if "email" in hint:
        em = fake.safe_email()
        return em, em
    if "phone" in hint or "tel" in hint:
        ph = _clean_phone(fake)
        return ph, ph
    if "county and state" in hint:
        cs = f"Travis County, {rng.choice(_US_STATES)}"
        return cs, cs
    if "city, state, and zip" in hint:
        csz = f"{fake.city()}, {rng.choice(_US_STATES)} {fake.random_int(10000, 99950):05d}"
        return csz, csz
    if "zip" in hint or "postal" in hint:
        zp = f"{fake.random_int(10000, 99950):05d}"
        return zp, zp
    if "state" in hint and ("incorporated" in hint or not f.max_len or f.max_len <= 2):
        st = rng.choice(_US_STATES)
        return st, st
    if "city" in hint:
        ct = fake.city()
        return ct, ct
    if "courtroom" in hint:
        return "Courtroom 4B", "Courtroom 4B"
    if "name" in name_only and "address" not in name_only and "organization" not in name_only:
        nm = fake.name()
        if f.max_len:
            nm = nm[: f.max_len]
        return nm, nm
    if "name and address" in hint or "name/address" in hint or "address" in hint or "street" in hint or "place" in hint or "location" in hint:
        addr = f"{_clean_street(fake)}, {fake.city()}, {rng.choice(_US_STATES)} {fake.random_int(10000, 99950):05d}"
        if "name" in hint:
            addr = f"{fake.name()}, {addr}"
        if f.max_len:
            addr = addr[: f.max_len]
        return addr, addr
    if "action number" in hint or "case number" in hint or "case_num" in hint or "docket" in hint:
        cn = f"1:26-cv-{fake.random_int(1000, 9999)}"
        return cn, cn
    if any(k in hint for k in ("fee", "amount", "cost", "price")):
        amt = f"{fake.random_int(15, 250)}.00"
        return amt, amt
    if any(k in hint for k in ("organization", "company", "employer", "corporation", "firm", "business", "legal name of entity")):
        co = fake.company()
        if f.max_len:
            co = co[: f.max_len]
        return co, co
    if "title" in hint and "name" not in hint:
        return "President", "President"
    if "first" in hint and "name" in hint:
        fn = fake.first_name()
        return fn, fn
    if "last" in hint and "name" in hint:
        ln = fake.last_name()
        return ln, ln
    if any(k in hint for k in ("name", "plaintiff", "defendant", "individual", "person", "served on", "left with", "subpoena_to", "party")):
        nm = fake.name()
        if f.max_len:
            nm = nm[: f.max_len]
        return nm, nm
    if any(k in hint for k in ("reason", "specify", "additional", "other", "description", "notes", "documents", "merchandise", "services provided")):
        phrase = "Consulting and software development services"
        if "document" in hint:
            phrase = " Employment contracts and billing records for 2025".strip()
        if f.max_len:
            phrase = phrase[: f.max_len]
        return phrase, phrase
    word = fake.word().capitalize()
    if f.max_len:
        word = word[: f.max_len]
    return word, word


# --------------------------------------------------------------------------- #
# Rendering source documents in 6 modalities
# --------------------------------------------------------------------------- #


def _load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        )
        if bold
        else (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        )
    )
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def render_scan_image(sample: GeneratedSample, *, rng: random.Random) -> Image.Image:
    width, height = 1200, 1550
    bg = rng.choice(((252, 251, 247), (250, 252, 255), (253, 250, 245)))
    img = Image.new("RGB", (width, height), color=bg)
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, width - 40, height - 40], outline=(205, 202, 195), width=2)
    f_title = _load_font(30, bold=True)
    f_sub = _load_font(19, bold=False)
    f_body = _load_font(21, bold=False)
    f_bold = _load_font(21, bold=True)

    y = 85
    draw.text((80, y), sample.title, fill=(25, 25, 30), font=f_title)
    y += 45
    draw.text((80, y), sample.subtitle, fill=(90, 90, 100), font=f_sub)
    y += 34
    draw.line([(80, y), (width - 80, y)], fill=(180, 180, 185), width=2)
    y += 30

    for line in sample.structured_lines:
        if ":" in line:
            head, rest = line.split(":", 1)
            draw.text((80, y), head + ":", fill=(20, 20, 25), font=f_bold)
            bbox = draw.textbbox((80, y), head + ": ", font=f_bold)
            avail_chars = max(28, int((width - 80 - bbox[2]) / 12.0))
            wrapped = textwrap.wrap(rest.strip(), width=avail_chars) or [""]
            draw.text((bbox[2], y), wrapped[0], fill=(30, 30, 35), font=f_body)
            y += 34
            for cont in wrapped[1:]:
                draw.text((110, y), cont, fill=(30, 30, 35), font=f_body)
                y += 34
        else:
            for cont in textwrap.wrap(line, width=78):
                draw.text((80, y), cont, fill=(30, 30, 35), font=f_body)
                y += 34
        y += 6
    return img


def write_source_file(sample: GeneratedSample, sources_dir: Path, *, rng: random.Random) -> str:
    """Writes the sample's unstructured source document in `sample.modality` and returns its relative path."""
    mod = sample.modality
    if mod == "txt":
        rel = "sources/intake_notes.txt"
        body = f"{sample.title}\n{sample.subtitle}\n\n" + "\n\n".join(sample.prose_paragraphs) + "\n"
        (sources_dir / "intake_notes.txt").write_text(body, encoding="utf-8")
        return rel

    if mod == "eml":
        rel = "sources/message.eml"
        eml = (
            f"From: {sample.sender_name} <{sample.sender_email}>\n"
            f"To: intake@acme-processing.example.com\n"
            f"Date: Wed, 07 Oct 2026 09:30:00 -0500\n"
            f"Subject: {sample.title} ({sample.sample_id})\n"
            f"MIME-Version: 1.0\n"
            f'Content-Type: text/plain; charset="utf-8"\n\n'
            + "\n\n".join(sample.prose_paragraphs)
            + f"\n\nBest regards,\n{sample.sender_name}\n"
        )
        (sources_dir / "message.eml").write_text(eml, encoding="utf-8")
        return rel

    if mod == "pdf":
        rel = "sources/document.pdf"
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=letter)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(64, 735, sample.title)
        c.setFont("Helvetica", 10)
        c.drawString(64, 718, sample.subtitle)
        y = 692
        for line in sample.structured_lines:
            for wrapped in textwrap.wrap(line, width=88):
                c.drawString(64, y, wrapped)
                y -= 16
        c.showPage()
        c.save()
        (sources_dir / "document.pdf").write_bytes(buf.getvalue())
        return rel

    img = render_scan_image(sample, rng=rng)
    if mod == "jpeg":
        rel = "sources/scan.jpg"
        img.save(sources_dir / "scan.jpg", format="JPEG", quality=90)
        return rel

    if mod == "png":
        rel = "sources/scan.png"
        img.save(sources_dir / "scan.png", format="PNG")
        return rel

    if mod == "scanned_pdf":
        rel = "sources/scanned_document.pdf"
        jpg_buf = io.BytesIO()
        img.save(jpg_buf, format="JPEG", quality=88)
        jpg_buf.seek(0)
        pdf_buf = io.BytesIO()
        c = canvas.Canvas(pdf_buf, pagesize=letter)
        c.drawImage(ImageReader(jpg_buf), 0, 0, width=letter[0], height=letter[1])
        c.showPage()
        c.save()
        (sources_dir / "scanned_document.pdf").write_bytes(pdf_buf.getvalue())
        return rel

    raise ValueError(f"Unknown modality: {mod!r}")


# --------------------------------------------------------------------------- #
# Dataset generation API + CLI
# --------------------------------------------------------------------------- #


def generate_dataset(
    output_dir: Path,
    *,
    template: str = "all",
    form_path: Path | None = None,
    count: int = 10,
    formats: Sequence[str] = FORMATS,
    seed: int = 42,
    flatten: bool = False,
    copy_form: bool = False,
) -> DatasetManifest:
    """Generates `count` fake filled PDFs, unstructured sources, and eval cases."""
    if count < 1:
        raise ValueError("count must be at least 1")
    valid_formats = tuple(f.strip().lower() for f in formats if f.strip())
    for fmt in valid_formats:
        if fmt not in FORMATS:
            raise ValueError(f"Unsupported format {fmt!r}; choose from {', '.join(FORMATS)}")

    output_dir = output_dir.resolve()
    filled_dir = output_dir / "filled"
    cases_dir = output_dir / "cases"
    filled_dir.mkdir(parents=True, exist_ok=True)
    cases_dir.mkdir(parents=True, exist_ok=True)

    fake = Faker("en_US")
    Faker.seed(seed)
    rng = random.Random(seed)
    settings = Settings()

    if form_path is not None:
        form_path = form_path.resolve()
        template_names = [ re.sub(r"[^A-Za-z0-9_-]", "_", form_path.stem) or "custom" ]
        template_paths = {template_names[0]: form_path}
    elif template == "all":
        template_names = list(TEMPLATES.keys())
        template_paths = dict(TEMPLATES)
    else:
        if template not in TEMPLATES:
            raise ValueError(f"Unknown template {template!r}; choose from {', '.join((*TEMPLATES, 'all'))}")
        template_names = [template]
        template_paths = {template: TEMPLATES[template]}

    loaded_forms: dict[str, LoadedForm] = {
        tname: open_form(tpath.read_bytes(), limits=settings.limits)
        for tname, tpath in template_paths.items()
    }
    form_bytes_by_name: dict[str, bytes] = {
        tname: tpath.read_bytes() for tname, tpath in template_paths.items()
    }

    filled_pdfs: list[Path] = []
    case_dirs: list[Path] = []
    width = max(4, len(str(count)))

    for idx in range(1, count + 1):
        tname = template_names[(idx - 1) % len(template_names)]
        modality = valid_formats[(idx - 1) % len(valid_formats)]
        sample_id = f"{tname}_{idx:0{width}d}"

        # Re-open a fresh LoadedForm for filling so pypdf writer state is clean.
        form = open_form(form_bytes_by_name[tname], limits=settings.limits)
        sample = _build_sample(sample_id, tname, modality, loaded_forms[tname], fake, rng)

        # 1. Write the filled ground-truth PDF.
        filled = fill_pdf(form, sample.pdf_values, flatten=flatten)
        filled_pdf_path = filled_dir / f"{sample_id}.pdf"
        filled_pdf_path.write_bytes(filled.pdf)
        filled_pdfs.append(filled_pdf_path)

        # 2. Write the eval case directory (`form.pdf`, `case.json`, `expected.json`, `sources/`).
        case_dir = cases_dir / sample_id
        sources_dir = case_dir / "sources"
        sources_dir.mkdir(parents=True, exist_ok=True)
        target_form = case_dir / "form.pdf"
        if target_form.exists() or target_form.is_symlink():
            target_form.unlink()
        if copy_form:
            target_form.write_bytes(form_bytes_by_name[tname])
        else:
            try:
                os.link(template_paths[tname], target_form)
            except OSError:
                try:
                    os.symlink(template_paths[tname], target_form)
                except OSError:
                    target_form.write_bytes(form_bytes_by_name[tname])

        rel_source = write_source_file(sample, sources_dir, rng=rng)
        case_spec = {
            "description": f"Synthetic {tname} record #{idx} ({modality}).",
            "today": sample.today,
            "sources": [{"role": "doc", "file": rel_source}],
        }
        (case_dir / "case.json").write_text(json.dumps(case_spec, indent=2) + "\n", encoding="utf-8")
        (case_dir / "expected.json").write_text(json.dumps(sample.expected, indent=2) + "\n", encoding="utf-8")
        case_dirs.append(case_dir)

    manifest_data = {
        "count": count,
        "seed": seed,
        "templates": template_names,
        "formats": list(valid_formats),
        "filled_dir": str(filled_dir),
        "cases_dir": str(cases_dir),
        "samples": [p.stem for p in filled_pdfs],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8")
    return DatasetManifest(
        output_dir=output_dir,
        count=count,
        seed=seed,
        templates=tuple(template_names),
        formats=valid_formats,
        filled_pdfs=tuple(filled_pdfs),
        case_dirs=tuple(case_dirs),
    )


def _build_sample(
    sample_id: str,
    tname: str,
    modality: str,
    form: LoadedForm,
    fake: Faker,
    rng: random.Random,
) -> GeneratedSample:
    if tname == "irs_w9":
        return _generate_irs_w9(sample_id, modality, form, fake, rng)
    if tname == "irs_w4":
        return _generate_irs_w4(sample_id, modality, form, fake, rng)
    if tname == "irs_8822":
        return _generate_irs_8822(sample_id, modality, form, fake, rng)
    if tname == "irs_w8ben":
        return _generate_irs_w8ben(sample_id, modality, form, fake, rng)
    if tname == "job_application":
        return _generate_job_application(sample_id, modality, form, fake, rng)
    return _generate_generic(sample_id, tname, modality, form, fake, rng)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m evals.dataset",
        description="Generate synthetic filled PDFs and multimodal eval cases using Faker.",
    )
    p.add_argument(
        "--template",
        choices=(*TEMPLATES.keys(), "all"),
        default="all",
        help="built-in form template to generate records for (default: all)",
    )
    p.add_argument(
        "--form",
        type=Path,
        default=None,
        help="custom AcroForm PDF path (overrides --template and synthesizes fields from its schema)",
    )
    p.add_argument(
        "-n",
        "--count",
        type=int,
        default=20,
        help="number of synthetic records / filled PDFs to generate (default: 20)",
    )
    p.add_argument(
        "--formats",
        default=",".join(FORMATS),
        help=f"comma-separated source formats to cycle through (default: {','.join(FORMATS)})",
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        required=True,
        help="output directory (writes filled/*.pdf, cases/*, and manifest.json)",
    )
    p.add_argument("--seed", type=int, default=42, help="random seed for reproducible generation (default: 42)")
    p.add_argument("--flatten", action="store_true", help="flatten the generated filled PDFs")
    p.add_argument("--copy-form", action="store_true", help="copy form.pdf into each case instead of symlinking")
    return p


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    out = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    manifest = generate_dataset(
        args.out,
        template=args.template,
        form_path=args.form,
        count=args.count,
        formats=formats,
        seed=args.seed,
        flatten=args.flatten,
        copy_form=args.copy_form,
    )
    print(
        f"Generated {manifest.count} fake filled PDF(s) and eval case(s) in {manifest.output_dir}\n"
        f"  Templates : {', '.join(manifest.templates)}\n"
        f"  Formats   : {', '.join(manifest.formats)}\n"
        f"  Filled PDF: {manifest.output_dir / 'filled'}\n"
        f"  Eval Cases: {manifest.output_dir / 'cases'}",
        file=out,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
