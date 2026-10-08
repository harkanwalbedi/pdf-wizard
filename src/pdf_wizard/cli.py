"""Command-line interface: `pdf-wizard inspect | schema | fill | serve`."""

from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import sys
import tempfile
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

from . import APP_NAME, __version__
from .config import MB, Settings
from .errors import InvalidInputError, PdfWizardError
from .mapping.base import Mapper
from .models import FieldResult, FieldStatus, FillReport, FormField, FormInfo

log = logging.getLogger(__name__)

MapperFactory = Callable[[Settings], Mapper]


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a whole number") from None
    if not 1 <= value <= 1_000_000:
        raise argparse.ArgumentTypeError("must be between 1 and 1000000")
    return value


def _unit_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1") from None
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1")
    return value


def _port(text: str) -> int:
    value = _positive_int(text)
    if not 1024 <= value <= 65535:
        raise argparse.ArgumentTypeError("must be between 1024 and 65535")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf-wizard",
        description=f"{APP_NAME} fills fillable (AcroForm) PDFs from your documents, with evidence for every value.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--debug", action="store_true", help="show tracebacks for unexpected errors (may include file contents)"
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    inspect = sub.add_parser("inspect", help="list a form's fields (no AI)")
    inspect.add_argument("form", type=Path, help="the PDF form")
    inspect.add_argument("--json", action="store_true", help="print JSON")

    schema = sub.add_parser(
        "schema",
        help="detect a form's schema (sections, checkbox mutex groups, split combs) and optionally enrich with AI",
    )
    schema.add_argument("form", type=Path, help="the PDF form")
    schema.add_argument("-o", "--output", type=Path, metavar="FILE", help="write the detected schema JSON to a file")
    schema.add_argument(
        "--enrich",
        action="store_true",
        help="use Gemini on the blank PDF (no user data) to enrich field labels, sections, and mutex groups",
    )
    schema.add_argument("--json", action="store_true", help="print schema JSON to stdout")
    schema.add_argument("--force", action="store_true", help="overwrite an existing output file")

    fill = sub.add_parser("fill", help="fill a form from your data and documents")
    fill.add_argument("form", type=Path, help="the PDF form to fill")
    fill.add_argument("-o", "--output", type=Path, help="where to write the filled PDF")
    fill.add_argument("--schema", type=Path, metavar="FILE",
                      help="use a pre-detected schema JSON file from 'pdf-wizard schema'")
    fill.add_argument("--enrich-schema", action="store_true",
                      help="run a visual AI schema-enrichment pass on the blank form before filling")
    fill.add_argument("--data", type=Path, action="append", default=[], metavar="FILE",
                      help="structured data: JSON or CSV (repeatable)")
    fill.add_argument("--row", type=_positive_int, metavar="N",
                      help="record to use when a data file has several (1 = first)")
    fill.add_argument("--doc", type=Path, action="append", default=[], metavar="FILE",
                      help="document: PDF, PNG, JPEG, plain text, or .eml (repeatable)")
    fill.add_argument("--reference", type=Path, action="append", default=[], metavar="FILE",
                      help="a previously filled PDF form about the same person (repeatable)")
    fill.add_argument("--note", action="append", default=[], metavar="TEXT",
                      help='a fact or preference in your own words, e.g. "Use my work address" (repeatable)')
    fill.add_argument("--report", type=Path, metavar="FILE", help="write a JSON report (sensitive values masked)")
    fill.add_argument("--flatten", action="store_true", help="produce a non-editable PDF")
    fill.add_argument("--min-confidence", type=_unit_float, metavar="X",
                      help="minimum confidence to fill a field (default 0.7)")
    fill.add_argument("--dry-run", action="store_true", help="map and check values but don't write a PDF")
    fill.add_argument("--force", action="store_true", help="overwrite existing output files")
    fill.add_argument("-q", "--quiet", action="store_true", help="print only the summary line")

    serve = sub.add_parser("serve", help="start the web UI on 127.0.0.1")
    serve.add_argument("--port", type=_port, default=8000, help="port (default 8000)")
    return parser


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    return run(argv)


def run(
    argv: Sequence[str] | None = None,
    *,
    mapper_factory: MapperFactory | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Runs a command and returns the exit code. Tests inject `mapper_factory`."""
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    args = build_parser().parse_args(argv)
    _setup_logging()
    try:
        settings = Settings.from_env()
        if args.command == "inspect":
            return _inspect(args, settings, out)
        if args.command == "schema":
            return _schema(args, settings, mapper_factory, out)
        if args.command == "fill":
            return _fill(args, settings, mapper_factory, out, err)
        return _serve(args, settings, mapper_factory, out)
    except PdfWizardError as exc:
        print(f"Error: {_safe(exc.user_message)}", file=err)
        return exc.exit_code
    except KeyboardInterrupt:
        print("Interrupted.", file=err)
        return 130
    except Exception as exc:  # last resort: never print values by default
        log.error("unexpected error: %s", type(exc).__name__)
        if args.debug:
            traceback.print_exc(file=err)
        else:
            print(f"Error: unexpected {type(exc).__name__}. Run with --debug for details.", file=err)
        return 1


def _setup_logging() -> None:
    from .logsafe import configure_logging

    configure_logging()
    logging.getLogger("pypdf").setLevel(logging.ERROR)  # noisy warnings about malformed PDFs
    # Warnings about malformed uploads can quote request header values (file names).
    for name in ("python_multipart", "multipart"):
        logging.getLogger(name).setLevel(logging.ERROR)


# --------------------------------------------------------------------------- #
# inspect
# --------------------------------------------------------------------------- #


def _inspect(args: argparse.Namespace, settings: Settings, out: TextIO) -> int:
    from .pipeline import inspect_form

    info = inspect_form(_read(args.form, settings), settings=settings)
    info = _mask_current_values(info)
    if args.json:
        print(json.dumps(info.model_dump(mode="json"), indent=2), file=out)
        return 0
    flags_doc = "R=required, RO=read-only, H=hidden, C=comb, M=multiline"
    print(f"{_safe(args.form.name)}: {len(info.fields)} fields on {info.page_count} page(s). Flags: {flags_doc}", file=out)
    rows = [("ID", "Type", "Page", "Flags", "Name", "Label", "Options / format")]
    for f in info.fields:
        flags = [n for n, on in (("R", f.required), ("RO", f.read_only), ("H", f.hidden), ("C", f.comb),
                                 ("M", f.multiline)) if on]
        if f.max_len:
            flags.append(f"max {f.max_len}")
        extra = ", ".join(o.value + (f" ({o.label})" if o.label else "") for o in f.options[:4])
        if len(f.options) > 4:
            extra += f", +{len(f.options) - 4} more"
        extra = extra or (f.format_hint or "")
        rows.append((f.id, f.type.value, str(f.page or "-"), " ".join(flags), f.name, f.label or "", extra))
    _print_table(rows, out, widths=(5, 9, 4, 14, 30, 30, 40))
    for warning in info.warnings:
        print(f"Warning: {_safe(warning)}", file=out)
    return 0


def _mask_current_values(info: FormInfo) -> FormInfo:
    return info.model_copy(update={"fields": _mask_fields_current_values(info.fields)})


def _mask_fields_current_values(fields: Sequence[FormField]) -> list[FormField]:
    from .privacy import is_sensitive, mask_value

    return [
        f.model_copy(update={"current_value": mask_value(f.current_value)})
        if f.current_value and is_sensitive(f, f.current_value)
        else f
        for f in fields
    ]


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #


def _schema(
    args: argparse.Namespace,
    settings: Settings,
    mapper_factory: MapperFactory | None,
    out: TextIO,
) -> int:
    from .pipeline import detect_form_schema

    if args.output is not None:
        _check_output(args.output, [args.form], args.force)
    mapper = None
    if args.enrich:
        mapper = mapper_factory(settings) if mapper_factory else _create_mapper(settings)
    form_bytes = _read(args.form, settings)
    schema = detect_form_schema(form_bytes, settings=settings, mapper=mapper, enrich=args.enrich)
    schema = schema.model_copy(update={"fields": _mask_fields_current_values(schema.fields)})

    if args.output is not None:
        _write_private(args.output, schema.model_dump_json(indent=2).encode("utf-8"), overwrite=args.force)

    if args.json:
        print(json.dumps(schema.model_dump(mode="json"), indent=2), file=out)
        return 0

    mode = f"enriched ({schema.model or 'model'})" if schema.enriched else "deterministic (no AI)"
    flags_doc = "R=required, RO=read-only, H=hidden, C=comb, M=multiline, OU=office-use"
    print(
        f"{_safe(args.form.name)}: {len(schema.fields)} fields on {schema.page_count} page(s), "
        f"{len(schema.mutex_groups)} mutex group(s) [{mode}]. Flags: {flags_doc}",
        file=out,
    )
    rows = [("ID", "Type", "Page", "Group", "Flags", "Section", "Label", "Options / format")]
    for f in schema.fields:
        flags = [
            n
            for n, on in (
                ("R", f.required),
                ("RO", f.read_only),
                ("H", f.hidden),
                ("C", f.comb),
                ("M", f.multiline),
                ("OU", f.office_use_only),
            )
            if on
        ]
        if f.max_len:
            flags.append(f"max {f.max_len}")
        extra = ", ".join(o.value + (f" ({o.label})" if o.label else "") for o in f.options[:4])
        if len(f.options) > 4:
            extra += f", +{len(f.options) - 4} more"
        extra = extra or (f.format_hint or "")
        rows.append((
            f.id,
            f.type.value,
            str(f.page or "-"),
            f.group or "",
            " ".join(flags),
            f.section or "",
            f.label or f.name,
            extra,
        ))
    _print_table(rows, out, widths=(5, 9, 4, 5, 14, 24, 34, 34))
    for warning in schema.warnings:
        print(f"Warning: {_safe(warning)}", file=out)
    if args.output is not None:
        print(f"Wrote {_safe(str(args.output))}", file=out)
    return 0


# --------------------------------------------------------------------------- #
# fill
# --------------------------------------------------------------------------- #


def _fill(args: argparse.Namespace, settings: Settings, mapper_factory: MapperFactory | None,
          out: TextIO, err: TextIO) -> int:
    from .pipeline import FillOptions, fill_form
    from .schema import parse_schema_json
    from .sources import SourceInput

    inputs_paths = [args.form, *([args.schema] if args.schema else []), *args.data, *args.doc, *args.reference]
    if not args.dry_run and args.output is None:
        raise InvalidInputError("Choose where to write the filled PDF with -o/--output (or use --dry-run).")
    if args.output is not None and not args.dry_run:
        _check_output(args.output, inputs_paths, args.force)
    if args.report is not None:
        _check_output(args.report, inputs_paths + ([args.output] if args.output else []), args.force)
    if not (args.data or args.doc or args.reference or args.note):
        raise InvalidInputError("Add at least one --data, --doc, --reference, or --note.")
    if args.schema is not None and args.enrich_schema:
        raise InvalidInputError("Use either --schema or --enrich-schema, not both.")

    # Fail fast on missing configuration, before reading files or calling the model.
    mapper = mapper_factory(settings) if mapper_factory else _create_mapper(settings)

    form_bytes = _read(args.form, settings)
    loaded_schema = (
        parse_schema_json(_read(args.schema, settings), max_bytes=settings.limits.max_file_bytes)
        if args.schema is not None
        else None
    )
    inputs = [SourceInput("data", p.name, _read(p, settings), row=args.row) for p in args.data]
    inputs += [SourceInput("doc", p.name, _read(p, settings)) for p in args.doc]
    inputs += [SourceInput("reference", p.name, _read(p, settings)) for p in args.reference]
    inputs += [SourceInput("note", "", note.encode("utf-8")) for note in args.note]

    options = FillOptions(
        flatten=args.flatten,
        dry_run=args.dry_run,
        min_confidence=args.min_confidence,
        schema=loaded_schema,
        enrich_schema=args.enrich_schema,
    )
    outcome = fill_form(form_bytes, inputs, mapper=mapper, settings=settings, options=options,
                        form_name=args.form.name)

    written: list[Path] = []
    if outcome.pdf is not None and args.output is not None:
        _write_private(args.output, outcome.pdf, overwrite=args.force)
        written.append(args.output)
    if args.report is not None:
        _write_private(args.report, outcome.report.model_dump_json(indent=2).encode("utf-8"), overwrite=args.force)
        written.append(args.report)
    _print_summary(outcome.report, written, out, quiet=args.quiet)
    return 0


def _create_mapper(settings: Settings) -> Mapper:
    from .mapping.gemini import create_mapper

    return create_mapper(settings)


def _read(path: Path, settings: Settings) -> bytes:
    limit = settings.limits.max_file_bytes
    try:
        if not path.is_file():
            raise InvalidInputError(f"Not a file: {_safe(str(path))}")
        if path.stat().st_size > limit:
            raise InvalidInputError(f"{_safe(path.name)} is too large (limit {limit // MB} MB).")
        with path.open("rb") as handle:
            data = handle.read(limit + 1)
    except OSError as exc:
        raise InvalidInputError(f"Can't read {_safe(str(path))}: {exc.strerror or 'error'}.") from exc
    if len(data) > limit:
        raise InvalidInputError(f"{_safe(path.name)} is too large (limit {limit // MB} MB).")
    return data


def _check_output(path: Path, inputs: list[Path], force: bool) -> None:
    resolved = path.resolve()
    if any(resolved == p.resolve() for p in inputs if p is not None):
        raise InvalidInputError(f"{_safe(str(path))} is also an input file. Choose another output name.")
    if path.is_dir():
        raise InvalidInputError(f"{_safe(str(path))} is a directory.")
    if path.exists() and not force:
        raise InvalidInputError(f"{_safe(str(path))} already exists. Use --force to overwrite it.")
    if not resolved.parent.is_dir():
        raise InvalidInputError(f"The folder for {_safe(str(path))} doesn't exist.")


def _write_private(path: Path, data: bytes, *, overwrite: bool) -> None:
    """Writes atomically with owner-only permissions: outputs contain personal data.

    Without `overwrite`, publishing fails if a file appeared at `path` after the
    up-front check (the model call takes a while), instead of replacing it.
    """
    try:
        fd, tmp = tempfile.mkstemp(dir=path.resolve().parent, prefix=".pdf-wizard-", suffix=".tmp")
    except OSError as exc:
        raise InvalidInputError(f"Can't write {_safe(str(path))}: {exc.strerror or 'error'}.") from exc
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o600)
        if overwrite:
            os.replace(tmp, path)
        else:
            _publish_new(tmp, path, data)
    except FileExistsError as exc:
        raise InvalidInputError(f"{_safe(str(path))} already exists. Use --force to overwrite it.") from exc
    except OSError as exc:
        raise InvalidInputError(f"Can't write {_safe(str(path))}: {exc.strerror or 'error'}.") from exc
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass  # renamed into place


def _publish_new(tmp: str, path: Path, data: bytes) -> None:
    """Puts the finished file at `path`, failing with FileExistsError if it exists."""
    try:
        os.link(tmp, path)  # unlike rename, never replaces an existing file
    except FileExistsError:
        raise
    except OSError:  # file systems without hard links: exclusive create (not atomic)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)


def _print_summary(report: FillReport, written: list[Path], out: TextIO, *, quiet: bool) -> None:
    counts = report.counts
    writable = sum(counts.get(s, 0) for s in (FieldStatus.FILLED, FieldStatus.NO_DATA, FieldStatus.REJECTED))
    verb = "would fill" if report.dry_run else "filled"
    print(
        f"{APP_NAME} {verb} {counts.get(FieldStatus.FILLED, 0)} of {writable} fields in {_safe(report.form_name)} "
        f"({counts.get(FieldStatus.NO_DATA, 0)} no data, {counts.get(FieldStatus.REJECTED, 0)} rejected).",
        file=out,
    )
    if quiet:
        return
    seconds = report.timings_ms.get("total", 0) / 1000
    print(f"Model: {report.model or '-'} · prompt {report.prompt_version or '-'} · {seconds:.1f} s", file=out)
    shown = [r for r in report.fields if r.status in (FieldStatus.FILLED, FieldStatus.REJECTED)]
    if shown:
        rows = [("ID", "Field", "Status", "Value", "Conf", "Source", "Note")]
        rows += [_result_row(r) for r in shown]
        print(file=out)
        _print_table(rows, out, widths=(5, 32, 8, 28, 4, 6, 50))
    for warning in report.warnings:
        print(f"Warning: {_safe(warning)}", file=out)
    for path in written:
        print(f"Wrote {_safe(str(path))}", file=out)


def _result_row(r: FieldResult) -> tuple[str, ...]:
    conf = f"{r.confidence:.2f}" if r.confidence is not None else ""
    note = r.detail or "; ".join(r.notes)
    return (r.field_id, r.label or r.name, r.status.value, r.value or "", conf, r.source_id or "", note or "")


# --------------------------------------------------------------------------- #
# serve
# --------------------------------------------------------------------------- #


def _serve(args: argparse.Namespace, settings: Settings, mapper_factory: MapperFactory | None, out: TextIO) -> int:
    import uvicorn

    from .web.app import create_app

    if mapper_factory is None:
        settings.validate_for_model_calls()  # fail closed at startup
        factory: Callable[[], Mapper] = lambda: _create_mapper(settings)  # noqa: E731
    else:
        factory = lambda: mapper_factory(settings)  # noqa: E731
    token = secrets.token_urlsafe(32)  # 256 bits; printed once, never logged
    app = create_app(settings, login_token=token, mapper_factory=factory)
    print(f"{APP_NAME} is running at http://127.0.0.1:{args.port}/", file=out)
    print(f"Sign-in token (valid until the server stops): {token}", file=out)
    print("Press Ctrl+C to stop. Jobs are kept in memory only and deleted when the server stops.", file=out)
    out.flush()
    # Loopback only: the web UI is a single-user local tool.
    uvicorn.run(app, host="127.0.0.1", port=args.port, server_header=False, proxy_headers=False,
                log_config=None, access_log=False)
    return 0


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #


def _safe(text: str) -> str:
    """Drops control characters so document text can't inject terminal escape sequences."""
    return "".join(ch if ch.isprintable() else " " for ch in str(text))


def _print_table(rows: list[tuple[str, ...]], out: TextIO, *, widths: tuple[int, ...]) -> None:
    for row in rows:
        cells = []
        for cell, width in zip(row, widths, strict=False):
            cell = _safe(cell)
            if len(cell) > width:
                cell = cell[: width - 1] + "…"
            cells.append(cell.ljust(width))
        print("  ".join(cells).rstrip(), file=out)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
