"""Data models shared by the pipeline, the CLI, and the web app.

Fields that can hold personal data (values, evidence, document text) are
excluded from `repr()` so that an accidental log line doesn't leak them.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Target form
# --------------------------------------------------------------------------- #


class FieldType(StrEnum):
    TEXT = "text"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    CHOICE = "choice"  # combo box or list box
    SIGNATURE = "signature"
    BUTTON = "button"  # push button


FILLABLE_TYPES = frozenset({FieldType.TEXT, FieldType.CHECKBOX, FieldType.RADIO, FieldType.CHOICE})


class Option(BaseModel):
    """An allowed value: a checkbox/radio on-state or a choice-field entry."""

    value: str  # export value; for buttons, the appearance state name without "/"
    label: str | None = None  # display text, or the text printed next to the widget


class FormField(BaseModel):
    id: str  # short ID used in prompts and reports: "F1", "F2", ...
    name: str  # fully qualified name: partial names (/T) joined with "."
    label: str | None = None  # tooltip (/TU), else the nearest text on the page
    type: FieldType
    page: int | None = None  # 1-based page of the first widget; None if there is no widget
    rect: tuple[float, float, float, float] | None = None  # first widget, PDF points
    options: list[Option] = Field(default_factory=list)
    max_len: int | None = None
    required: bool = False
    read_only: bool = False
    multiline: bool = False
    comb: bool = False
    editable: bool = False  # combo box that also accepts free text
    multi_select: bool = False
    hidden: bool = False
    password: bool = False
    file_select: bool = False
    # Parsed from Acrobat format scripts (never executed). One of:
    # "date:<acrobat format>", "zip", "zip4", "phone", "ssn",
    # "number:<decimals>:<sep style>", "percent:<decimals>:<sep style>", "mask:<mask>"
    format_hint: str | None = None
    section: str | None = None  # section/part heading on the form
    group: str | None = None  # mutually-exclusive checkbox group ID, e.g. "G1"
    office_use_only: bool = False  # official/agency use only; never filled from user sources
    current_value: str | None = Field(default=None, repr=False)

    @property
    def writable(self) -> bool:
        """True if PDF Wizard may write this field (it is offered to the model)."""
        return self.type in FILLABLE_TYPES and not (
            self.read_only or self.hidden or self.password or self.file_select
        )

    @property
    def on_states(self) -> list[str]:
        """Checkbox/radio "on" appearance states (everything except Off)."""
        return [o.value for o in self.options if o.value != "Off"]


class FormInfo(BaseModel):
    sha256: str
    page_count: int
    fields: list[FormField]
    had_xfa: bool = False  # hybrid AcroForm + XFA; XFA is dropped when filling
    encrypted: bool = False  # encrypted with an empty user password
    permissions: int | None = None  # /P flags when encrypted
    warnings: list[str] = Field(default_factory=list)

    def field(self, field_id: str) -> FormField | None:
        return next((f for f in self.fields if f.id == field_id), None)


class FormSchema(BaseModel):
    """Detected schema for a blank form, reusable across fills of the same PDF."""

    schema_version: str = "1"
    form_sha256: str
    page_count: int
    enriched: bool = False  # True when visual LLM schema enrichment was applied
    model: str | None = None
    fields: list[FormField]
    mutex_groups: dict[str, list[str]] = Field(default_factory=dict)  # group ID -> checkbox field IDs
    warnings: list[str] = Field(default_factory=list)


class SchemaFieldEnrichment(BaseModel):
    """Visual schema enrichment for one field returned by Gemini."""

    field_id: str = Field(description="ID of the form field from the catalog, e.g. 'F4'.")
    label: str | None = Field(
        default=None,
        description="Clear human-readable label for this field as printed on the form (include line/item number if any).",
    )
    section: str | None = Field(
        default=None,
        description="Section or part heading that this field belongs to, e.g. 'Part I: Taxpayer Identification'.",
    )
    format_hint: str | None = Field(
        default=None,
        description="Optional format hint: 'date:mm/dd/yyyy', 'date:yyyy-mm-dd', 'ssn', 'zip', 'zip4', 'phone', or None.",
    )
    mutex_group: str | None = Field(
        default=None,
        description="For checkboxes where only one box in a set may be checked, a shared group name (e.g. 'G1').",
    )
    office_use_only: bool = Field(
        default=False,
        description="True if the form marks this field as 'For office/official/employer/agency use only'.",
    )


class SchemaEnrichmentResponse(BaseModel):
    fields: list[SchemaFieldEnrichment] = Field(
        description="Enriched metadata for the form's fields."
    )


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #


class SourceKind(StrEnum):
    SYSTEM = "system"  # S0: facts supplied by PDF Wizard (today's date)
    STRUCTURED = "structured"  # JSON / CSV record
    TEXT = "text"
    EMAIL = "email"
    IMAGE = "image"
    PDF = "pdf"
    REFERENCE_FORM = "reference_form"  # a previously filled form about the same person
    NOTE = "note"  # free text typed by the user


class Source(BaseModel):
    id: str  # "S0", "S1", ...
    kind: SourceKind
    name: str  # display name: sanitized file name or "Note 1"
    text: str | None = Field(default=None, repr=False)  # text sent to the model
    # Text used only to verify evidence (e.g. a PDF's text layer when only the
    # file is sent to the model). Falls back to `text`.
    check_text: str | None = Field(default=None, repr=False)
    keypaths: dict[str, str] = Field(default_factory=dict, repr=False)  # structured: "a.b[0]" -> value
    media_type: str | None = None  # MIME type of `data`: application/pdf, image/png, image/jpeg
    data: bytes | None = Field(default=None, repr=False, exclude=True)  # file sent to the model

    @property
    def verification_text(self) -> str:
        return self.check_text if self.check_text is not None else (self.text or "")

    @property
    def verifiable(self) -> bool:
        """False for images and scanned PDFs: code can't check evidence against them."""
        return bool(self.keypaths) or bool(self.verification_text.strip())


# --------------------------------------------------------------------------- #
# Mapping (model output)
# --------------------------------------------------------------------------- #


class FieldProposal(BaseModel):
    """One value proposed by the model. This class is the Gemini response schema."""

    field_id: str = Field(description="ID of the form field from the field catalog, e.g. 'F12'.")
    value: str = Field(
        repr=False,
        description=(
            "Value to enter. Checkbox: 'checked' or 'unchecked'. Radio button or choice field: "
            "one of the field's option values, exactly as listed. Text: formatted to match the "
            "field's format and no longer than its max_len."
        ),
    )
    source_id: str = Field(description="ID of the source the value comes from, e.g. 'S2'.")
    evidence: str = Field(
        repr=False,
        description=(
            "Short verbatim quote from that source that supports the value. For structured "
            "data, the key path (e.g. 'applicant.address.city')."
        ),
    )
    confidence: float = Field(
        description="Probability from 0.0 to 1.0 that this value is correct for this field."
    )
    sensitive: bool = Field(
        description=(
            "True if the value is sensitive personal data: government or tax ID, account or "
            "card number, date of birth, password, health data."
        )
    )


class MappingResponse(BaseModel):
    fields: list[FieldProposal] = Field(
        description="One entry per field you can fill from the sources. Omit all other fields."
    )


class MappingResult(BaseModel):
    proposals: list[FieldProposal]
    model: str
    prompt_version: str
    usage: dict[str, int] = Field(default_factory=dict)  # token counts


# --------------------------------------------------------------------------- #
# Results and report
# --------------------------------------------------------------------------- #


class FieldStatus(StrEnum):
    FILLED = "filled"
    NO_DATA = "no_data"  # no usable proposal; left as is
    REJECTED = "rejected"  # a proposal failed a guardrail; left as is
    NOT_WRITABLE = "not_writable"  # read-only, hidden, signature, button...


class RejectReason(StrEnum):
    INVALID_VALUE = "invalid_value"  # type, option, length or format check failed
    UNVERIFIED = "unverified"  # evidence not found in the cited source
    LOW_CONFIDENCE = "low_confidence"
    UNKNOWN_SOURCE = "unknown_source"
    RENDER_FAILED = "render_failed"  # the PDF library couldn't write the value


class FieldResult(BaseModel):
    field_id: str
    name: str
    label: str | None = None
    type: FieldType
    page: int | None = None
    status: FieldStatus
    # Display value: the text, option value, or "checked"/"unchecked". For a
    # rejected field, the rejected proposal. Masked in reports when sensitive.
    value: str | None = Field(default=None, repr=False)
    source_id: str | None = None
    evidence: str | None = Field(default=None, repr=False)
    confidence: float | None = None
    sensitive: bool = False
    reason: RejectReason | None = None
    detail: str | None = None  # safe explanation; never contains values
    notes: list[str] = Field(default_factory=list)  # e.g. conflicting proposals


class SourceSummary(BaseModel):
    id: str
    kind: SourceKind
    name: str
    chars: int = 0  # characters of text sent to the model
    has_file: bool = False  # a file part (PDF/image) was sent to the model
    verifiable: bool = True


class FillReport(BaseModel):
    tool: str = "pdf-wizard"
    tool_version: str
    created_at: str  # ISO 8601, UTC
    form_name: str
    form_sha256: str
    page_count: int
    model: str | None = None
    prompt_version: str | None = None
    flattened: bool = False
    dry_run: bool = False
    masked: bool = False  # True once sensitive values and evidence are masked
    counts: dict[str, int] = Field(default_factory=dict)  # by FieldStatus
    fields: list[FieldResult] = Field(default_factory=list)
    sources: list[SourceSummary] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    timings_ms: dict[str, int] = Field(default_factory=dict)
    usage: dict[str, int] = Field(default_factory=dict)
