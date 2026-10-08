"""Email documents (RFC 822 / .eml): selected headers, then the text body."""

from __future__ import annotations

import codecs
import email
import email.header
import email.policy
import encodings
import encodings.aliases
import logging
import re
from email.message import EmailMessage
from html.parser import HTMLParser
from typing import cast

from ..errors import InvalidInputError, PdfWizardError
from ..models import Source, SourceKind
from .common import Context, clean_text, invalid

log = logging.getLogger(__name__)

SHOWN_HEADERS = ("From", "To", "Cc", "Date", "Subject")
_SHOWN_KEYS = frozenset(h.lower() for h in SHOWN_HEADERS)
_MAX_HEADER_CHARS = 2000
_HTML_EXPANSION = 10  # HTML longer than 10x max_source_chars is cut before conversion

# The stdlib parser re-parses Content-* headers on every access, at a cost that
# grows faster than linearly with their length, and checks every line against
# each open multipart boundary. These caps keep a hostile email cheap to parse;
# real emails stay far below them.
_MAX_CONTENT_HEADER_BYTES = 2048
_MAX_CONTENT_HEADERS_TOTAL_BYTES = 32 * 1024
_MAX_CONTAINERS = 20  # multipart/* and message/* parts
_MAX_BOUNDARY_LINES = 2000  # lines starting with "--": bounds the number of parts

_CONTENT_HEADER = re.compile(rb"[\r\n](content-[a-z0-9-]*)[ \t]*:", re.I)
_LOGICAL_LINE_END = re.compile(rb"\r\n(?![ \t])|\r(?![\n \t])|\n(?![ \t])")  # not a folded line
_CONTAINER_TYPE = re.compile(rb"multipart|message/", re.I)
_BOUNDARY_LINE = re.compile(rb"[\r\n]--")

# The sender picks the charset, and with it the codec. Only these real text
# encodings are used (modules of the encodings package, once aliases such as
# "latin1" or "windows-31j" are resolved); anything else is decoded as UTF-8.
# punycode and idna are registered text codecs with quadratic decoders (a 10 MB
# body would take about 28 minutes of CPU).
_TEXT_CODEC_MODULES = frozenset(
    [
        *"ascii utf_8 utf_8_sig utf_16 utf_16_le utf_16_be utf_32 utf_32_le utf_32_be latin_1".split(),
        *(f"iso8859_{n}" for n in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16)),
        *(f"cp125{n}" for n in range(9)),  # windows-1250 to windows-1258
        *"gb2312 gbk gb18030 big5 big5hkscs cp950 shift_jis cp932 euc_jp euc_kr cp949".split(),
        *"iso2022_jp iso2022_jp_1 iso2022_jp_2 iso2022_jp_2004 iso2022_jp_3 iso2022_jp_ext".split(),
        *"koi8_r koi8_u mac_roman".split(),
    ]
)
_CHARSET_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,39}")  # RFC 2978: at most 40 characters

# Content-based detection (files without a .eml name).
_SNIFF_BYTES = 64 * 1024
_LINE_BREAK = re.compile(rb"\r\n|\r|\n")
_FIRST_HEADER = re.compile(rb"[A-Za-z][A-Za-z0-9-]*:[ \t]")
_HEADER_NAME = re.compile(rb"([!-9;-~]+):")
# Besides From/Subject, a MIME or transport header is required, so that memos
# and "Key: value" text files stay plain text instead of losing every line
# that isn't one of SHOWN_HEADERS.
_TRANSPORT_HEADERS = frozenset(
    {
        b"mime-version",
        b"content-type",
        b"content-transfer-encoding",
        b"message-id",
        b"received",
        b"return-path",
        b"delivered-to",
        b"dkim-signature",
        b"authentication-results",
        b"x-mailer",
        b"user-agent",
        b"in-reply-to",
        b"references",
    }
)


def looks_like_email(data: bytes) -> bool:
    """True if `data` starts with an RFC 822 header block of a real email."""
    lines = _LINE_BREAK.split(data[:_SNIFF_BYTES])
    if len(data) > _SNIFF_BYTES:
        lines.pop()  # possibly cut in the middle
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines) and lines[i].startswith(b"From "):  # mbox separator line
        i += 1
    if i >= len(lines) or not _FIRST_HEADER.match(lines[i]):
        return False
    names: set[bytes] = set()
    for line in lines[i:]:
        if not line.strip():
            break
        if line[:1] in (b" ", b"\t"):  # folded continuation
            continue
        match = _HEADER_NAME.match(line)
        if match is None:
            return False  # not a header block
        names.add(match.group(1).lower())
    return bool(names & {b"from", b"subject"}) and bool(names & _TRANSPORT_HEADERS)


def load_email(data: bytes, ctx: Context) -> Source:
    """From/To/Cc/Date/Subject lines, a blank line, then the plain-text (or HTML) body."""
    data = data.lstrip(b" \t\r\n")
    _check_complexity(data, ctx)
    try:
        msg = cast(EmailMessage, email.message_from_bytes(data, policy=email.policy.default))
        headers = _header_lines(msg)
        body_part = msg.get_body(preferencelist=("plain", "html"))
        body = _body_text(body_part, ctx) if body_part is not None else ""
        attachments = _has_attachments(msg, body_part)
    except PdfWizardError:
        raise
    except Exception as exc:  # malformed MIME, RecursionError, ...
        log.info("email could not be parsed: source=%s error=%s", ctx.sid, type(exc).__name__)
        raise invalid(ctx, "could not be read as an email") from exc

    if attachments:
        ctx.warn("email attachments were ignored; upload them as separate documents.")
    text = "\n".join(headers)
    if body:
        text = f"{text}\n\n{body}" if text else body
    if not text:
        raise invalid(ctx, "has no readable text")
    return Source(id=ctx.sid, kind=SourceKind.EMAIL, name=ctx.name, text=ctx.truncate(text))


def _check_complexity(data: bytes, ctx: Context) -> None:
    scan = b"\n" + data  # every header line now follows a line break
    total = containers = 0
    for match in _CONTENT_HEADER.finditer(scan):
        start = match.start(1)
        end_match = _LOGICAL_LINE_END.search(scan, start, start + _MAX_CONTENT_HEADER_BYTES + 3)
        end = end_match.start() if end_match else len(scan)
        total += end - start
        if end - start > _MAX_CONTENT_HEADER_BYTES or total > _MAX_CONTENT_HEADERS_TOTAL_BYTES:
            raise _too_complex(ctx)
        if match.group(1).lower() == b"content-type" and _CONTAINER_TYPE.search(scan, match.end(), end):
            containers += 1
            if containers > _MAX_CONTAINERS:
                raise _too_complex(ctx)
    for count, _ in enumerate(_BOUNDARY_LINE.finditer(scan), start=1):
        if count > _MAX_BOUNDARY_LINES:
            raise _too_complex(ctx)


def _too_complex(ctx: Context) -> InvalidInputError:
    return invalid(ctx, "has too many parts or headers to be read as an email")


def _header_lines(msg: EmailMessage) -> list[str]:
    # Raw values are decoded here instead of through policy.default's header
    # classes: its address parser is quadratic on hostile input.
    found: dict[str, list[str]] = {}
    sizes: dict[str, int] = {}
    for name, raw in msg.raw_items():
        key = name.strip().lower()
        if key not in _SHOWN_KEYS or sizes.get(key, 0) >= _MAX_HEADER_CHARS:
            continue
        text = str(raw)
        # Raw sizes count, so that repeated headers that decode to nothing
        # (e.g. empty encoded words) still use up the budget.
        sizes[key] = sizes.get(key, 0) + len(text) + 1
        if value := _decode_header(text):
            found.setdefault(key, []).append(value)
    return [
        f"{label}: {', '.join(found[label.lower()])[:_MAX_HEADER_CHARS]}"
        for label in SHOWN_HEADERS
        if label.lower() in found
    ]


def _decode_header(raw: str) -> str:
    value = raw[: _MAX_HEADER_CHARS * 2]  # bounds the decoder's (super-linear) work
    try:  # undo the parser's surrogateescape decoding of raw 8-bit bytes
        value = value.encode("utf-8", "surrogateescape").decode("utf-8", "replace")
    except UnicodeError:
        value = value.encode("utf-8", "replace").decode("utf-8")
    value = re.sub(r"\r\n|\r|\n", "", value)  # unfold
    # Splitting on whitespace also removes U+2028 and the other line breaks.
    return " ".join(clean_text(_decode_words(value)).split())[:_MAX_HEADER_CHARS]


def _decode_words(value: str) -> str:
    """Decodes RFC 2047 encoded words ("=?iso-8859-1?q?caf=E9?=") with allowed charsets only."""
    try:
        parts = email.header.decode_header(value)
    except Exception:  # malformed encoded words: keep the raw text
        return value
    words: list[str] = []
    for part, charset in parts:
        if isinstance(part, str):  # the header has no encoded words
            words.append(part)
        elif charset is None:  # text between encoded words, returned as raw-unicode-escape bytes
            words.append(part.decode("raw-unicode-escape", errors="replace"))
        else:  # drop an RFC 2231 language suffix: "iso-8859-1*fr"
            words.append(decode_declared(part, charset.partition("*")[0]))
    return "".join(words)


def text_codec(charset: str | None) -> str:
    """The codec for a charset named in an email: the charset if it's an allowed text encoding, else UTF-8."""
    name = (charset or "").strip().lower()
    if not _CHARSET_NAME.fullmatch(name):
        return "utf-8"
    # Aliases are resolved as codecs.lookup() resolves them, so that names outside
    # the allow-list never reach the codec registry (each miss tries an import).
    norm = encodings.normalize_encoding(name)
    aliases = encodings.aliases.aliases
    module = aliases.get(norm) or aliases.get(norm.replace(".", "_")) or norm
    if module not in _TEXT_CODEC_MODULES:
        return "utf-8"
    try:
        return codecs.lookup(module).name
    except LookupError:  # not expected: every allowed codec ships with Python
        return "utf-8"


def decode_declared(data: bytes, charset: str | None) -> str:
    """Decodes bytes in the charset the sender named (see text_codec); bad bytes become U+FFFD."""
    return data.decode(text_codec(charset), errors="replace")


def _body_text(part: EmailMessage, ctx: Context) -> str:
    payload = part.get_payload(decode=True)
    if not isinstance(payload, bytes):
        return ""
    text = decode_declared(payload, part.get_content_charset())
    if part.get_content_subtype() == "html":
        limit = ctx.limits.max_source_chars * _HTML_EXPANSION
        if len(text) > limit:
            text = text[:limit]
            ctx.mark_truncated()
        text = html_to_text(text)
    return clean_text(text).strip()


def _has_attachments(msg: EmailMessage, body: EmailMessage | None) -> bool:
    if not msg.is_multipart():
        return body is None  # the whole message is one non-text part
    if any(True for _ in msg.iter_attachments()):
        return True
    return any(part.is_attachment() for part in msg.walk())


# --------------------------------------------------------------------------- #
# HTML to text
# --------------------------------------------------------------------------- #

_WHITESPACE = re.compile(r"\s+")


class _HtmlToText(HTMLParser):
    """Collects visible text. Never fetches, renders, or executes anything."""

    SKIP = frozenset({"script", "style"})
    LINE_BREAK = frozenset({"br", "hr"})  # newline on the start tag only
    BLOCK = frozenset(
        "address article aside blockquote dd div dl dt figcaption figure footer form h1 h2 h3 h4 "
        "h5 h6 header li main nav ol p pre section table tbody tfoot thead title tr ul".split()
    )
    CELL = frozenset({"td", "th"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0
        self._pre = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP:
            self._skip += 1
        elif tag in self.LINE_BREAK or tag in self.BLOCK:
            self.parts.append("\n")
        elif tag in self.CELL:
            self.parts.append(" ")
        if tag == "pre":
            self._pre += 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.LINE_BREAK or tag in self.BLOCK:  # <br/>; a self-closed <script/> is empty
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag in self.BLOCK:
            self.parts.append("\n")
        elif tag in self.CELL:
            self.parts.append(" ")
        if tag == "pre":
            self._pre = max(0, self._pre - 1)

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data if self._pre else _WHITESPACE.sub(" ", data))


def html_to_text(html: str) -> str:
    """Visible text of an HTML document; block elements become line breaks."""
    parser = _HtmlToText()
    parser.feed(html)
    parser.close()
    lines = (" ".join(line.split()) for line in "".join(parser.parts).split("\n"))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
