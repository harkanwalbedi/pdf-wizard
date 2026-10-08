"""Exception types.

Every error carries a `user_message` that is safe to show to the user: it never
contains field values, document contents, or internal details. Diagnostic
details belong in the exception chain (`raise ... from exc`) and in logs.
"""


class PdfWizardError(Exception):
    """Base class for expected, user-facing failures."""

    exit_code = 1

    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


class InvalidInputError(PdfWizardError):
    """An input file or option is invalid (wrong type, too large, malformed)."""

    exit_code = 2


class UnsupportedFormError(PdfWizardError):
    """The target PDF can't be filled (no AcroForm fields, XFA-only, locked...)."""

    exit_code = 3


class MappingError(PdfWizardError):
    """The AI backend failed or returned an unusable response."""

    exit_code = 4


class ConfigError(PdfWizardError):
    """Required configuration (project, credentials, model) is missing or invalid."""

    exit_code = 4
