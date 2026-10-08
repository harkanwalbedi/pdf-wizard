"""Security tests of the web UI (local, single-user mode), with no network.

Covers response headers, the Host and method filters, body caps, sessions and
their cookie, CSRF, rate limits, authorization, error pages, and static checks
of the templates and front-end code. Helpers and fixtures come from test_web.py.
"""

from __future__ import annotations

import ast
import base64
import json
import logging
import re
from collections.abc import Iterator, Sequence

import pytest

from pdf_wizard.config import Limits, Settings
from pdf_wizard.mapping.fake import FakeMapper
from pdf_wizard.web.app import (
    LOGIN_EXPIRED,
    LOGIN_NO_COOKIE,
    SESSION_COOKIE_HTTP,
    STATIC_DIR,
    TEMPLATES_DIR,
    WEB_DIR,
    create_app,
)
from pdf_wizard.web.jobs import JobStore
from pdf_wizard.web.security import (
    RateLimiter,
    SessionStore,
    client_key,
    display_name,
    download_name,
    is_allowed_upload,
    is_cross_site_request,
    parse_origin,
    retry_after,
    server_origin,
)
from test_web import (  # noqa: F401 - app, client, clock, and stub are pytest fixtures
    BASE_URL,
    JPEG_BYTES,
    JSON_BYTES,
    PDF_BYTES,
    PNG_BYTES,
    SESSION_COOKIE,
    TOKEN,
    FakeClock,
    StubFill,
    TestClient,
    app,
    build_app,
    client,
    clock,
    cookie_payload,
    csrf_from,
    esc,
    job_id_of,
    multipart,
    open_client,
    sign_in,
    stub,
    upload,
    wait_for_job,
)

CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; font-src 'self'; "
    "connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
)
SECURITY_HEADERS = {
    "content-security-policy": CSP,
    "x-frame-options": "DENY",
    "x-content-type-options": "nosniff",
    # Not "no-referrer": with it, browsers send `Origin: null` on the app's own posts.
    "referrer-policy": "same-origin",
    "permissions-policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
    "cross-origin-opener-policy": "same-origin",
    "cross-origin-resource-policy": "same-origin",
}
UNKNOWN_JOB = "A" * 22
SMALL_BODY_LIMIT = 64 * 1024
SMALL_LIMITS = Limits(max_file_bytes=2_000, max_total_bytes=5_000, max_source_files=2)
UPLOAD_BODY_LIMIT = SMALL_LIMITS.max_total_bytes + 1024 * 1024
FORM_URLENCODED = {"content-type": "application/x-www-form-urlencoded"}

# Fetch Metadata and Origin headers as browsers send them. The test client sends
# none of these by default, like scripts and older browsers.
TYPED_URL = {"sec-fetch-site": "none", "sec-fetch-mode": "navigate", "sec-fetch-dest": "document"}
OWN_FORM_POST = {"origin": BASE_URL, "sec-fetch-site": "same-origin", "sec-fetch-mode": "navigate",
                 "sec-fetch-dest": "document"}
BLIND_POST = {"sec-fetch-site": "cross-site", "sec-fetch-mode": "no-cors", "sec-fetch-dest": "empty"}
# Posts that another site, or a page on another loopback port, makes through the user's browser.
CROSS_SITE_POSTS = {
    "no-cors-fetch": BLIND_POST,
    "form-post": {"origin": "https://evil.example", "sec-fetch-site": "cross-site", "sec-fetch-mode": "navigate",
                  "sec-fetch-dest": "document"},
    "other-port-same-site": {"origin": "https://localhost:8443", "sec-fetch-site": "same-site"},
    "other-site-origin-only": {"origin": "https://evil.example"},  # browsers without Fetch Metadata
    "null-origin": {"origin": "null"},  # sandboxed frames, file:// pages, some redirects
    "other-scheme": {"origin": "http://localhost"},
    "other-port": {"origin": "https://localhost:8443"},
    "other-loopback-name": {"origin": "https://127.0.0.1"},  # localhost and 127.0.0.1 are different origins
    "lookalike-host": {"origin": "https://localhost.evil.example"},
    "malformed-origin": {"origin": "localhost"},
    "same-origin-site-with-foreign-origin": {"origin": "https://evil.example", "sec-fetch-site": "same-origin"},
    "own-origin-but-cross-site": {"origin": BASE_URL, "sec-fetch-site": "cross-site"},
    "unknown-site-value": {"sec-fetch-site": "somewhere"},
}
# (Sec-Fetch-Mode, Sec-Fetch-Dest) of loads that aren't top-level navigations.
SUBRESOURCE_LOADS = (("no-cors", "image"), ("no-cors", "script"), ("no-cors", "style"), ("no-cors", "empty"),
                     ("cors", "empty"), ("navigate", "iframe"))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def assert_secure_headers(response, *, cache: str = "no-store") -> None:
    where = f"{response.request.method} {response.request.url.path} -> {response.status_code}"
    for name, value in SECURITY_HEADERS.items():
        assert response.headers.get(name) == value, f"{name} on {where}"
    assert response.headers.get("cache-control") == cache, where
    assert not any(name.startswith("access-control-") for name in response.headers), where


def cookie_attributes(response, name: str = SESSION_COOKIE) -> tuple[str, dict[str, str]]:
    """(value, {lower-cased attribute: value}) of the one Set-Cookie header for `name`."""
    headers = [h for h in response.headers.get_list("set-cookie") if h.startswith(f"{name}=")]
    assert len(headers) == 1, response.headers.get_list("set-cookie")
    first, *rest = (part.strip() for part in headers[0].split(";"))
    attributes = {}
    for part in rest:
        key, _, value = part.partition("=")
        attributes[key.lower()] = value
    return first.split("=", 1)[1], attributes


def get_with_cookie(client: TestClient, path: str, cookie: str, *, name: str = SESSION_COOKIE):
    """GET with exactly this session cookie (the client's cookie jar is cleared)."""
    client.cookies.clear()
    return client.get(path, headers={"cookie": f"{name}={cookie}"}, follow_redirects=False)


def other_browser(app) -> TestClient:
    """A second client without its own lifespan (fine for requests that start no jobs)."""
    return TestClient(app, base_url=BASE_URL)


def finished_job(client: TestClient, csrf: str) -> str:
    job_id = job_id_of(upload(client, csrf))
    wait_for_job(client, job_id)
    return job_id


def post_forms(html: str) -> list[tuple[str, str]]:
    forms = re.findall(r"<form\b([^>]*)>(.*?)</form>", html, re.S)
    assert all('method="post"' in attributes for attributes, _ in forms), "every form must use POST"
    return forms


def http_scope(method: str = "POST", *, scheme: str = "https", host: str | None = "localhost",
               headers: Sequence[tuple[str, str]] = ()) -> dict:
    """A minimal ASGI HTTP scope, for unit tests of the cross-site check."""
    raw = [] if host is None else [(b"host", host.encode())]
    raw += [(name.encode(), value.encode()) for name, value in headers]
    return {"type": "http", "method": method, "scheme": scheme, "path": "/login", "headers": raw}


# --------------------------------------------------------------------------- #
# Response headers
# --------------------------------------------------------------------------- #


def test_security_headers_on_every_kind_of_response(client: TestClient):
    responses = [
        client.get("/login"),  # page
        client.get("/", follow_redirects=False),  # redirect
        client.get("/no-such-page"),  # 404 page
        client.get(f"/jobs/{UNKNOWN_JOB}/status"),  # 401 JSON
        client.put("/"),  # 405 from the method filter
        client.get("/login", headers={"host": "evil.example"}),  # 400 from the host filter
        client.post("/login", data={"token": TOKEN}, follow_redirects=False),  # 403: no CSRF token
        client.post("/login", content=b"x", headers={**FORM_URLENCODED, "content-length": str(10**9)}),  # 413
        client.post("/login", data={"token": TOKEN}, headers=BLIND_POST),  # 403 from the cross-site filter
    ]
    csrf = sign_in(client)
    job_id = finished_job(client, csrf)
    responses += [
        client.get("/"),
        client.get(f"/jobs/{job_id}"),
        client.get(f"/jobs/{job_id}/status"),  # JSON
        client.get(f"/jobs/{job_id}/download"),  # PDF
        client.post("/jobs", data={"csrf_token": "wrong"}),  # 403
    ]
    assert [r.status_code for r in responses] == [200, 303, 404, 401, 405, 400, 403, 413, 403,
                                                  200, 200, 200, 200, 403]
    for response in responses:
        assert_secure_headers(response)


@pytest.mark.parametrize(
    ("path", "content_type"),
    [
        ("/static/app.css", "text/css"),
        ("/static/app.js", "javascript"),
        ("/static/logo.svg", "image/svg+xml"),
        ("/static/favicon.svg", "image/svg+xml"),
    ],
)
def test_static_files_have_security_headers(client: TestClient, path: str, content_type: str):
    response = client.get(path)
    assert response.status_code == 200
    assert content_type in response.headers["content-type"]  # must be right: nosniff is on
    assert_secure_headers(response, cache="no-cache")


def test_server_errors_show_a_generic_page(client: TestClient, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG)
    secret = "boom /home/jane/tax 123-45-6789"
    sign_in(client)

    def broken(owner: str):
        raise RuntimeError(secret)

    client.app.state.web.jobs.list_for = broken
    response = client.get("/")
    assert response.status_code == 500
    assert "Something went wrong" in response.text and secret not in response.text
    assert_secure_headers(response)
    assert "RuntimeError" in caplog.text and secret not in caplog.text


def test_error_pages_never_echo_request_data(client: TestClient):
    probe = "<script>alert(1)</script>"
    job_probe = "<svg onload=alert(1)>"  # no "/", so these requests reach the job routes
    responses = [
        client.get(f"/{probe}"),
        client.get(f"/jobs/{job_probe}/status"),
        client.request("PROPFIND", f"/{probe}"),
        client.get("/login", headers={"host": "alert1.example"}),
    ]
    csrf = sign_in(client)
    responses += [
        client.get(f"/jobs/{job_probe}"),
        client.get(f"/jobs/{job_probe}/status"),
        client.get(f"/jobs/{job_probe}/download"),
        client.post(f"/jobs/{job_probe}/delete", data={"csrf_token": csrf}),
        client.post("/jobs", data={"csrf_token": probe}),
    ]
    assert [r.status_code for r in responses] == [404, 401, 405, 400, 404, 404, 404, 404, 403]
    for response in responses:
        assert "alert(1)" not in response.text and "alert1" not in response.text
        assert "PROPFIND" not in response.text


# --------------------------------------------------------------------------- #
# Host and method filters, disabled API docs, static files
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("host", ["evil.example", "localhost.evil.example", "127.0.0.1.nip.io", "127.0.0.2",
                                  "0.0.0.0", "[::1]"])
def test_other_hosts_are_rejected(client: TestClient, host: str):
    # Blocks DNS rebinding: a page on evil.example resolving to 127.0.0.1 sends Host: evil.example.
    response = client.get("/login", headers={"host": host})
    assert response.status_code == 400
    assert "set-cookie" not in response.headers
    assert_secure_headers(response)


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "localhost:8765", "127.0.0.1:8765"])
def test_loopback_hosts_are_served(client: TestClient, host: str):
    assert client.get("/login", headers={"host": host}).status_code == 200


@pytest.mark.parametrize("method", ["PUT", "DELETE", "PATCH", "OPTIONS", "TRACE", "HEAD", "PROPFIND"])
def test_only_get_and_post_are_allowed(client: TestClient, method: str):
    for path in ("/", "/login", "/logout", "/jobs", f"/jobs/{UNKNOWN_JOB}", f"/jobs/{UNKNOWN_JOB}/status",
                 f"/jobs/{UNKNOWN_JOB}/download", "/static/app.css", "/no-such-page"):
        response = client.request(method, path)
        assert response.status_code == 405, (method, path)
        assert response.headers["allow"] == "GET, POST"
        assert_secure_headers(response, cache="no-cache" if path.startswith("/static/") else "no-store")


def test_no_cross_origin_access(client: TestClient):
    preflight = client.options(
        "/jobs", headers={"origin": "https://evil.example", "access-control-request-method": "POST"}
    )
    assert preflight.status_code == 405
    response = client.get("/login", headers={"origin": "https://evil.example"})
    for result in (preflight, response):
        assert not any(name.startswith("access-control-") for name in result.headers)


def test_routes_accept_only_their_own_method(client: TestClient):
    csrf = sign_in(client)
    job_id = finished_job(client, csrf)
    for method, path in (("GET", "/logout"), ("GET", "/jobs"), ("GET", f"/jobs/{job_id}/delete"),
                         ("POST", "/"), ("POST", f"/jobs/{job_id}"), ("POST", f"/jobs/{job_id}/status"),
                         ("POST", f"/jobs/{job_id}/download")):
        data = {"csrf_token": csrf} if method == "POST" else None
        assert client.request(method, path, data=data).status_code == 405, (method, path)
    assert client.get("/").status_code == 200  # still signed in
    assert client.get(f"/jobs/{job_id}").status_code == 200  # not deleted


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"])
def test_api_docs_are_disabled(client: TestClient, path: str):
    response = client.get(path)
    assert response.status_code == 404
    assert "swagger" not in response.text.lower() and "openapi" not in response.text.lower()


@pytest.mark.parametrize("path", ["/static/%2e%2e/app.py", "/static/..%2fapp.py", "/static/%2e%2e/templates/base.html",
                                  "/static/", "/static/missing.js"])
def test_static_files_stay_in_their_directory(client: TestClient, path: str):
    response = client.get(path)
    assert response.status_code == 404
    assert "create_app" not in response.text and "{%" not in response.text


# --------------------------------------------------------------------------- #
# Cross-site requests (Fetch Metadata and Origin)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("headers", list(CROSS_SITE_POSTS.values()), ids=list(CROSS_SITE_POSTS))
def test_cross_site_posts_are_rejected(client: TestClient, stub: StubFill, headers: dict[str, str]):
    csrf = sign_in(client)
    job_id = finished_job(client, csrf)
    responses = [
        client.post("/login", data={"csrf_token": csrf, "token": TOKEN}, headers=headers, follow_redirects=False),
        upload(client, csrf, headers=headers),
        client.post(f"/jobs/{job_id}/delete", data={"csrf_token": csrf}, headers=headers, follow_redirects=False),
        client.post("/logout", data={"csrf_token": csrf}, headers=headers, follow_redirects=False),
    ]
    for response in responses:
        path = response.request.url.path
        assert response.status_code == 403 and "That request was blocked" in response.text, path
        assert "set-cookie" not in response.headers, path  # rejected before the session middleware
        assert_secure_headers(response)
    assert len(stub.calls) == 1  # no new job
    assert client.get(f"/jobs/{job_id}").status_code == 200  # not deleted, and still signed in


@pytest.mark.parametrize(
    "headers",
    [OWN_FORM_POST, {"origin": BASE_URL}, {"origin": "https://localhost:443"}, {"sec-fetch-site": "none"}, {}],
    ids=["same-origin", "origin-only", "default-port-spelled-out", "user-started", "no-headers"],
)
def test_posts_from_this_origin_are_accepted(client: TestClient, stub: StubFill, headers: dict[str, str]):
    page = client.get("/login", headers=TYPED_URL)
    response = client.post("/login", data={"csrf_token": csrf_from(page.text), "token": TOKEN}, headers=headers,
                           follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/"
    csrf = csrf_from(client.get("/").text)
    job_id = job_id_of(upload(client, csrf, headers=headers))
    wait_for_job(client, job_id)
    response = client.post(f"/jobs/{job_id}/delete", data={"csrf_token": csrf}, headers=headers,
                           follow_redirects=False)
    assert response.status_code == 303 and client.get(f"/jobs/{job_id}").status_code == 404
    response = client.post("/logout", data={"csrf_token": csrf}, headers=headers, follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/login"
    assert len(stub.calls) == 1


def test_origin_must_match_the_address_and_port_in_use(clock: FakeClock):
    app = build_app(fill_fn=StubFill(), clock=clock)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        page = client.get("/login")
        cookie, _ = cookie_attributes(page, SESSION_COOKIE_HTTP)
        form = {"csrf_token": csrf_from(page.text), "token": TOKEN}

        def sign_in_from(origin: str):
            # httpx (unlike browsers on loopback) won't send Secure cookies over http://.
            headers = {"cookie": f"{SESSION_COOKIE_HTTP}={cookie}", "origin": origin}
            return client.post("/login", data=form, headers=headers, follow_redirects=False)

        # A page on another loopback port is same-site, but it is another origin.
        for origin in ("http://127.0.0.1:9999", "http://127.0.0.1", "http://localhost:8765", "https://127.0.0.1:8765"):
            assert sign_in_from(origin).status_code == 403, origin
        assert sign_in_from("http://127.0.0.1:8765").status_code == 303


def test_cross_site_requests_never_use_up_the_users_limits(client: TestClient, app):
    stranger = other_browser(app)  # the user's browser on another site's page: no cookie (SameSite=Strict)
    image = {"sec-fetch-site": "cross-site", "sec-fetch-mode": "no-cors", "sec-fetch-dest": "image"}
    for _ in range(160):  # 320 requests: more than the per-client limit of 300 a minute
        assert stranger.post("/login", data={"token": "guess"}, headers=BLIND_POST).status_code == 403
        assert stranger.get("/login", headers=image).status_code == 403
    for origin in ("null", "https://evil.example"):  # browsers without Fetch Metadata
        for _ in range(5):
            assert stranger.post("/login", data={"token": "guess"}, headers={"origin": origin}).status_code == 403
    sign_in(client)  # neither the per-client limit nor the sign-in limit was touched


@pytest.mark.parametrize("site", ["cross-site", "same-site"])
def test_other_sites_cant_load_the_app_as_a_subresource(client: TestClient, site: str):
    csrf = sign_in(client)
    job_id = finished_job(client, csrf)
    for path in ("/", "/login", f"/jobs/{job_id}", f"/jobs/{job_id}/status", f"/jobs/{job_id}/download",
                 "/static/app.css", "/no-such-page"):
        for mode, dest in SUBRESOURCE_LOADS:
            response = client.get(path, headers={"sec-fetch-site": site, "sec-fetch-mode": mode, "sec-fetch-dest": dest})
            assert response.status_code == 403, (path, mode, dest)
            assert "set-cookie" not in response.headers
            assert_secure_headers(response, cache="no-cache" if path.startswith("/static/") else "no-store")
    assert client.get("/").status_code == 200  # still signed in


def test_own_requests_and_links_from_other_pages_are_allowed(client: TestClient):
    assert client.get("/login", headers=TYPED_URL).status_code == 200  # address bar, bookmark
    csrf = sign_in(client)
    job_id = finished_job(client, csrf)
    own_fetch = {"sec-fetch-site": "same-origin", "sec-fetch-mode": "cors", "sec-fetch-dest": "empty"}
    own_script = {"sec-fetch-site": "same-origin", "sec-fetch-mode": "no-cors", "sec-fetch-dest": "script"}
    assert client.get(f"/jobs/{job_id}/status", headers=own_fetch).status_code == 200
    assert client.get("/static/app.js", headers=own_script).status_code == 200
    for site in ("cross-site", "same-site"):  # following a link is a top-level navigation
        link = {"sec-fetch-site": site, "sec-fetch-mode": "navigate", "sec-fetch-dest": "document"}
        assert client.get(f"/jobs/{job_id}", headers=link).status_code == 200


def test_origin_parsing():
    assert parse_origin("https://localhost") == ("https", "localhost", 443)
    assert parse_origin("http://127.0.0.1:8765") == ("http", "127.0.0.1", 8765)
    assert parse_origin("HTTP://LocalHost:80") == ("http", "localhost", 80)
    for bad in ("null", "", "localhost", "https://", "file:///etc/passwd", "ftp://localhost", "https://localhost:0",
                "https://localhost:65536", "https://localhost:8765:1", "https://user@localhost", "https://localhost/",
                "https://local host", "https://localhost:abc"):
        assert parse_origin(bad) is None, bad
    assert server_origin(http_scope(host="localhost")) == ("https", "localhost", 443)
    assert server_origin(http_scope(scheme="http", host="127.0.0.1:8765")) == ("http", "127.0.0.1", 8765)
    assert server_origin(http_scope(host=None)) is None
    assert server_origin(http_scope(headers=[("host", "localhost")])) is None  # two Host headers
    assert server_origin(http_scope(host="localhost:abc")) is None
    assert server_origin(http_scope(scheme="ws", host="localhost")) is None


def test_cross_site_check_edge_cases():
    own = ("origin", "https://localhost")
    assert not is_cross_site_request(http_scope())  # no headers: scripts, older browsers (CSRF still applies)
    assert not is_cross_site_request(http_scope(headers=[own, ("sec-fetch-site", "same-origin")]))
    assert not is_cross_site_request(http_scope(headers=[("sec-fetch-site", "Same-Origin")]))
    assert not is_cross_site_request(http_scope(headers=[("sec-fetch-site", "none")]))
    # Repeated headers are never trusted.
    assert is_cross_site_request(http_scope(headers=[own, own]))
    assert is_cross_site_request(http_scope(headers=[("sec-fetch-site", "same-origin")] * 2))
    assert is_cross_site_request(http_scope("GET", headers=[("sec-fetch-site", "same-origin")] * 2))
    # When this server's own origin can't be worked out, no Origin header matches it.
    assert is_cross_site_request(http_scope(host="localhost:abc", headers=[("origin", "https://localhost:abc")]))
    assert is_cross_site_request(http_scope(host=None, headers=[own]))
    # Safe methods: other sites may only navigate to the app, as a top-level document.
    mode, dest = ("sec-fetch-mode", "navigate"), ("sec-fetch-dest", "document")
    assert not is_cross_site_request(http_scope("GET", headers=[("sec-fetch-site", "cross-site"), mode, dest]))
    assert is_cross_site_request(http_scope("GET", headers=[("sec-fetch-site", "cross-site"), mode]))
    assert is_cross_site_request(http_scope("GET", headers=[("sec-fetch-site", "same-site"), dest]))
    assert not is_cross_site_request(http_scope("GET", headers=[("origin", "https://evil.example")]))


# --------------------------------------------------------------------------- #
# Request body caps
# --------------------------------------------------------------------------- #


def test_declared_body_size_is_checked_before_the_app_runs(clock: FakeClock):
    stub = StubFill()
    app = build_app(fill_fn=stub, clock=clock, settings=Settings(limits=SMALL_LIMITS))
    with open_client(app) as client:
        csrf = sign_in(client)
        response = client.post("/jobs", content=b"--x--\r\n", headers={
            "content-type": "multipart/form-data; boundary=x", "content-length": str(UPLOAD_BODY_LIMIT + 1)})
        assert response.status_code == 413 and "Files too large" in response.text
        # Every other route takes a small body only.
        for path in ("/login", "/logout", f"/jobs/{UNKNOWN_JOB}/delete"):
            response = client.post(path, content=b"a=b",
                                   headers={**FORM_URLENCODED, "content-length": str(SMALL_BODY_LIMIT + 1)})
            assert response.status_code == 413 and "Request too large" in response.text, path
        for bad in ("abc", "-1", "1e9", "12 34", "9" * 40):
            response = client.post("/login", content=b"a=b", headers={**FORM_URLENCODED, "content-length": bad})
            assert response.status_code == (413 if bad.isdigit() else 400), bad
        assert stub.calls == []
        assert client.get("/").status_code == 200  # /logout never ran


def test_streamed_body_size_is_counted(clock: FakeClock):
    stub = StubFill()
    app = build_app(fill_fn=stub, clock=clock, settings=Settings(limits=SMALL_LIMITS))

    def chunks(size: int) -> Iterator[bytes]:
        yield b"a" * size  # a generator body is sent chunked, without Content-Length

    with open_client(app) as client:
        client.get("/login")
        response = client.post("/login", content=chunks(SMALL_BODY_LIMIT + 1), headers=FORM_URLENCODED)
        assert response.status_code == 413
        sign_in(client)
        response = client.post("/jobs", content=chunks(UPLOAD_BODY_LIMIT + 1),
                               headers={"content-type": "multipart/form-data; boundary=x"})
        assert response.status_code == 413 and "Files too large" in response.text
        assert stub.calls == []


def test_sign_in_form_is_parsed_with_small_limits(client: TestClient):
    csrf = csrf_from(client.get("/login").text)
    for parts in (
        [("csrf_token", None, csrf.encode()), ("token", "token.txt", TOKEN.encode())],  # no files
        [("csrf_token", None, csrf.encode())] + [(f"field{i}", None, b"x") for i in range(9)],  # too many fields
        [("csrf_token", None, csrf.encode()), ("token", None, b"x" * 5_000)],  # a field over 4 KB
    ):
        body, headers = multipart(parts)
        response = client.post("/login", content=body, headers=headers)
        assert response.status_code == 400
        assert "Too many" not in response.text and "exceeded" not in response.text  # no parser details
    for token in (TOKEN + "x" * 600, "ü" * 40):
        response = client.post("/login", data={"csrf_token": csrf, "token": token})
        assert response.status_code == 401
    assert client.get("/", follow_redirects=False).status_code == 303


# --------------------------------------------------------------------------- #
# Sessions and the session cookie
# --------------------------------------------------------------------------- #


def test_session_cookie_attributes(client: TestClient):
    page = client.get("/login")
    _, before = cookie_attributes(page)
    response = client.post("/login", data={"csrf_token": csrf_from(page.text), "token": TOKEN},
                           follow_redirects=False)
    _, after = cookie_attributes(response)
    assert SESSION_COOKIE == "__Host-pdf_wizard_session"
    for attributes in (before, after):
        assert "secure" in attributes and "httponly" in attributes
        assert attributes["samesite"].lower() == "strict"
        assert attributes["path"] == "/"
        assert "domain" not in attributes
        assert attributes["max-age"] == str(8 * 60 * 60)


def test_plain_http_loopback_drops_only_the_host_prefix(clock: FakeClock):
    # `pdf-wizard serve` uses http://127.0.0.1, where Chromium rejects '__Host-' cookies.
    app = build_app(fill_fn=StubFill(), clock=clock)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        page = client.get("/login")
        assert not any(h.startswith("__Host-") for h in page.headers.get_list("set-cookie"))
        cookie, attributes = cookie_attributes(page, SESSION_COOKIE_HTTP)
        assert SESSION_COOKIE_HTTP == "pdf_wizard_session"
        assert "secure" in attributes and "httponly" in attributes
        assert attributes["samesite"].lower() == "strict" and attributes["path"] == "/"
        assert "domain" not in attributes
        # httpx (unlike browsers on loopback) won't send Secure cookies over http://.
        response = client.post("/login", data={"csrf_token": csrf_from(page.text), "token": TOKEN},
                               headers={"cookie": f"{SESSION_COOKIE_HTTP}={cookie}"}, follow_redirects=False)
        assert response.status_code == 303
        signed_in, _ = cookie_attributes(response, SESSION_COOKIE_HTTP)
        assert get_with_cookie(client, "/", signed_in, name=SESSION_COOKIE_HTTP).status_code == 200


def test_cookie_holds_only_random_ids(client: TestClient, app):
    page = client.get("/login")
    assert cookie_payload(client.cookies[SESSION_COOKIE]) == {"login_csrf": csrf_from(page.text)}
    csrf = sign_in(client)
    payload = cookie_payload(client.cookies[SESSION_COOKIE])
    assert set(payload) == {"sid"}
    sid = payload["sid"]
    assert len(sid) >= 43 and sid != csrf
    assert app.state.web.sessions.get(sid).csrf_token == csrf  # the CSRF token stays on the server


def test_sign_in_rotates_the_session(client: TestClient, app):
    page = client.get("/login")
    login_csrf = csrf_from(page.text)
    anonymous_cookie = client.cookies[SESSION_COOKIE]
    first_csrf = sign_in(client)
    first_cookie = client.cookies[SESSION_COOKIE]
    first_sid = cookie_payload(first_cookie)["sid"]
    assert first_cookie != anonymous_cookie and first_csrf != login_csrf

    client.post("/logout", data={"csrf_token": first_csrf})
    second_csrf = sign_in(client)
    second_sid = cookie_payload(client.cookies[SESSION_COOKIE])["sid"]
    assert second_sid != first_sid and second_csrf != first_csrf
    assert len(app.state.web.sessions) == 1

    # Neither a pre-sign-in cookie (e.g. one planted before the victim signs in)
    # nor the old session's cookie gives access.
    for cookie in (anonymous_cookie, first_cookie):
        response = get_with_cookie(client, "/", cookie)
        assert response.status_code == 303 and response.headers["location"] == "/login"


def test_forged_and_foreign_cookies_are_ignored(client: TestClient, clock: FakeClock):
    other_app = build_app(fill_fn=StubFill(), clock=clock)  # same token, its own session key
    sign_in(client)
    cookie = client.cookies[SESSION_COOKIE]
    other = other_browser(other_app)
    sign_in(other)
    foreign = other.cookies[SESSION_COOKIE]
    _, signature = cookie.split(".", 1)
    forged = base64.b64encode(json.dumps({"sid": "x" * 43}).encode()).decode()
    unsigned = base64.b64encode(json.dumps(cookie_payload(cookie)).encode()).decode()
    for bad in (foreign, f"{forged}.{signature}", unsigned, "null", ""):
        response = get_with_cookie(client, "/", bad)
        assert response.status_code == 303 and response.headers["location"] == "/login", bad
    assert get_with_cookie(client, "/", cookie).status_code == 200  # the real one still works


def test_session_cap_ends_the_least_recently_used_session(client: TestClient, clock: FakeClock):
    csrf = sign_in(client)
    finished_job(client, csrf)
    cookies = [client.cookies[SESSION_COOKIE]]
    for _ in range(20):  # successful sign-ins don't count toward the sign-in limit
        clock.advance(1)  # a distinct last-use time for each session
        client.cookies.clear()
        sign_in(client)
        cookies.append(client.cookies[SESSION_COOKIE])
    web = client.app.state.web
    assert len(web.sessions) == 20
    assert len(web.jobs) == 0  # the ended session's job went with it
    assert get_with_cookie(client, "/", cookies[0]).status_code == 303
    for cookie in cookies[1:]:
        assert get_with_cookie(client, "/", cookie).status_code == 200


def test_logout_needs_post_and_csrf_and_clears_the_cookie(client: TestClient):
    csrf = sign_in(client)
    assert client.get("/logout").status_code == 405
    assert client.post("/logout", data={}).status_code == 403
    assert client.get("/").status_code == 200  # still signed in
    response = client.post("/logout", data={"csrf_token": csrf}, follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/login"
    value, attributes = cookie_attributes(response)
    assert value == "null" and "1970" in attributes["expires"]
    # Browsers only accept the deletion of a __Host- cookie with these attributes.
    assert "secure" in attributes and attributes["path"] == "/" and "domain" not in attributes


def test_session_store_expiry_cap_and_end_hook():
    clock = FakeClock()
    ended: list[str] = []
    store = SessionStore(clock=clock, idle_s=1800, absolute_s=8 * 3600, max_sessions=2, on_end=ended.append)
    first = store.create()
    assert first.sid != first.csrf_token and len(first.sid) >= 43
    assert first.sid not in repr(first) and first.csrf_token not in repr(first)
    clock.advance(1801)
    assert store.get(first.sid) is None and ended == [first.sid]
    second, third = store.create(), store.create()
    fourth = store.create()  # over the cap: the least recently used session ends
    assert ended[-1] == second.sid and store.get(second.sid) is None
    third.authenticated = False
    assert store.get(third.sid) is None
    assert store.get(fourth.sid) is fourth


# --------------------------------------------------------------------------- #
# Authorization
# --------------------------------------------------------------------------- #


def test_protected_routes_need_a_session(client: TestClient, app):
    csrf = sign_in(client)
    job_id = finished_job(client, csrf)
    anonymous = other_browser(app)
    anonymous.get("/login")  # a pre-sign-in session: only a sign-in CSRF token
    for method, path in (("GET", "/"), ("GET", f"/jobs/{job_id}"), ("GET", f"/jobs/{job_id}/download"),
                         ("POST", "/jobs"), ("POST", f"/jobs/{job_id}/delete"), ("POST", "/logout")):
        data = {"csrf_token": csrf} if method == "POST" else None
        response = anonymous.request(method, path, data=data, follow_redirects=False)
        assert response.status_code == 303 and response.headers["location"] == "/login", (method, path)
    response = anonymous.get(f"/jobs/{job_id}/status")
    assert response.status_code == 401
    assert response.headers["content-type"] == "application/json"
    assert set(response.json()) == {"error"}
    assert_secure_headers(response)
    assert client.get(f"/jobs/{job_id}").status_code == 200  # untouched


def test_job_store_checks_the_owner_on_every_lookup():
    store = JobStore(clock=FakeClock(), ttl_s=3600, max_per_owner=20, max_total=100)
    job = store.create("alice-session", "a.pdf")
    assert re.fullmatch(r"[A-Za-z0-9_-]{22}", job.id)  # 128 random bits
    assert store.get(job.id, "alice-session") is job
    for intruder in ("mallory-session", "", "alice-session ", "ALICE-SESSION"):
        assert store.get(job.id, intruder) is None
        assert store.delete(job.id, intruder) is False
        assert store.list_for(intruder) == []
    assert store.get(job.id, "alice-session") is job
    assert store.delete(job.id, "alice-session") is True and len(store) == 0


def test_status_api_returns_only_status_and_counts(client: TestClient):
    csrf = sign_in(client)
    response = client.get(f"/jobs/{finished_job(client, csrf)}/status")
    assert response.headers["content-type"] == "application/json"
    assert set(response.json()) == {"status", "counts"}
    assert_secure_headers(response)


# --------------------------------------------------------------------------- #
# CSRF
# --------------------------------------------------------------------------- #


def test_sign_in_requires_the_csrf_token(client: TestClient, app):
    login_csrf = csrf_from(client.get("/login").text)
    other_csrf = csrf_from(other_browser(app).get("/login").text)
    for fields in ({}, {"csrf_token": ""}, {"csrf_token": "wrong"}, {"csrf_token": other_csrf},
                   {"csrf_token": login_csrf + " "}):
        response = client.post("/login", data={"token": TOKEN, **fields}, follow_redirects=False)
        assert response.status_code == 403, fields
        assert TOKEN not in response.text
    assert client.get("/", follow_redirects=False).status_code == 303  # never signed in


def test_sign_in_csrf_failures_explain_what_to_do(client: TestClient):
    # No cookie at all: the browser dropped or blocked it.
    response = client.post("/login", data={"token": TOKEN, "csrf_token": "x" * 43}, follow_redirects=False)
    assert response.status_code == 403 and esc(LOGIN_NO_COOKIE) in response.text
    # A cookie that doesn't match the page (a stale page, or the server restarted).
    response = client.post("/login", data={"token": TOKEN, "csrf_token": "stale"}, follow_redirects=False)
    assert response.status_code == 403 and esc(LOGIN_EXPIRED) in response.text
    # The page shown has a working token.
    response = client.post("/login", data={"token": TOKEN, "csrf_token": csrf_from(response.text)},
                           follow_redirects=False)
    assert response.status_code == 303


def test_every_post_rejects_a_missing_or_wrong_csrf_token(client: TestClient, app, clock: FakeClock,
                                                          stub: StubFill):
    login_csrf = csrf_from(client.get("/login").text)
    csrf = sign_in(client)
    job_id = finished_job(client, csrf)
    other_csrf = sign_in(other_browser(app))
    for bad in (None, "", "wrong", login_csrf, other_csrf, csrf + " ", csrf.upper()):
        clock.advance(61)  # stay under the upload rate limit, which is checked first
        fields = {} if bad is None else {"csrf_token": bad}
        responses = [
            upload(client, bad),
            client.post(f"/jobs/{job_id}/delete", data=fields, follow_redirects=False),
            client.post("/logout", data=fields, follow_redirects=False),
        ]
        for response in responses:
            assert response.status_code == 403, (bad, response.request.url.path)
            assert "That request was blocked" in response.text
    assert len(stub.calls) == 1 and len(app.state.web.jobs) == 1  # no new job
    assert client.get(f"/jobs/{job_id}").status_code == 200  # not deleted
    assert client.get("/").status_code == 200  # still signed in


def test_every_post_form_carries_the_csrf_token(client: TestClient):
    login_page = client.get("/login").text
    [(attributes, body)] = post_forms(login_page)
    assert f'<input type="hidden" name="csrf_token" value="{csrf_from(login_page)}">' in body
    csrf = sign_in(client)
    job_id = finished_job(client, csrf)
    pages = {"index": client.get("/").text, "job": client.get(f"/jobs/{job_id}").text}
    for name, html in pages.items():
        forms = post_forms(html)
        assert len(forms) == 2, name  # upload or delete, plus sign out
        for attributes, body in forms:
            assert re.search(r'\saction="/[^"]*"', attributes), attributes  # same-origin targets only
            assert f'<input type="hidden" name="csrf_token" value="{csrf}">' in body, (name, attributes)


# --------------------------------------------------------------------------- #
# Rate limits
# --------------------------------------------------------------------------- #


def test_sign_in_rate_limit(client: TestClient, clock: FakeClock):
    csrf = csrf_from(client.get("/login").text)
    for _ in range(5):
        response = client.post("/login", data={"csrf_token": csrf, "token": "not-the-token-" + "x" * 30})
        assert response.status_code == 401
    # While the failed attempts are used up, no token is checked, not even the right one.
    response = client.post("/login", data={"csrf_token": csrf, "token": TOKEN}, follow_redirects=False)
    assert response.status_code == 429
    assert 1 <= int(response.headers["retry-after"]) <= 60
    assert_secure_headers(response)
    assert client.get("/", follow_redirects=False).status_code == 303  # not signed in
    assert client.post("/login", data={"token": TOKEN}, follow_redirects=False).status_code == 403  # CSRF first
    clock.advance(61)
    response = client.post("/login", data={"csrf_token": csrf, "token": TOKEN}, follow_redirects=False)
    assert response.status_code == 303


def test_only_failed_tokens_count_toward_the_sign_in_limit(client: TestClient, app):
    for _ in range(10):  # successful sign-ins aren't counted
        client.cookies.clear()
        sign_in(client)
    stranger = other_browser(app)  # the same client address, like every local browser tab and process
    for _ in range(20):  # no cookie and no CSRF token: what a page on another site can send blind
        stranger.cookies.clear()
        assert stranger.post("/login", data={"token": "guess"}, follow_redirects=False).status_code == 403
    csrf = csrf_from(stranger.get("/login").text)
    for _ in range(20):  # a wrong CSRF token
        response = stranger.post("/login", data={"csrf_token": "wrong", "token": "guess"}, follow_redirects=False)
        assert response.status_code == 403
    for _ in range(5):  # wrong tokens that pass the CSRF check are counted
        response = stranger.post("/login", data={"csrf_token": csrf, "token": "guess"}, follow_redirects=False)
        assert response.status_code == 401
    response = stranger.post("/login", data={"csrf_token": csrf, "token": "guess"}, follow_redirects=False)
    assert response.status_code == 429


def test_upload_rate_limit(client: TestClient, clock: FakeClock):
    csrf = sign_in(client)
    for _ in range(6):
        assert upload(client, csrf).status_code == 303
    response = upload(client, csrf)
    assert response.status_code == 429 and 1 <= int(response.headers["retry-after"]) <= 60
    assert upload(client, None).status_code == 429  # checked before the body and the CSRF token
    clock.advance(61)
    assert upload(client, csrf).status_code == 303


def test_signed_out_requests_are_limited_per_client(client: TestClient, app, clock: FakeClock):
    sign_in(client)
    clock.advance(61)  # a fresh window
    stranger = other_browser(app)  # no session: counted per client
    for _ in range(301):  # static files are never counted
        assert stranger.get("/static/favicon.svg").status_code == 200
    for _ in range(300):
        assert stranger.get("/", follow_redirects=False).status_code == 303
    response = stranger.get("/login")
    assert response.status_code == 429 and "Too many requests" in response.text
    assert 1 <= int(response.headers["retry-after"]) <= 60
    assert_secure_headers(response)
    assert stranger.get("/static/app.css").status_code == 200
    assert stranger.get(f"/jobs/{UNKNOWN_JOB}/status").status_code == 401  # the status API has its own budget
    assert client.get("/").status_code == 200  # a signed-in session has its own budget
    clock.advance(61)
    assert stranger.get("/login").status_code == 200


def test_signed_in_requests_are_limited_per_session(client: TestClient, app, clock: FakeClock):
    csrf = sign_in(client)
    job_id = finished_job(client, csrf)
    other = other_browser(app)
    sign_in(other)
    clock.advance(61)  # a fresh window
    for _ in range(300):
        assert client.get("/login", follow_redirects=False).status_code == 303  # signed in: on to /
    response = client.get("/")
    assert response.status_code == 429 and "Too many requests" in response.text
    assert 1 <= int(response.headers["retry-after"]) <= 60
    assert_secure_headers(response)
    assert client.get(f"/jobs/{job_id}/status").status_code == 200  # polling has its own budget
    assert other.get("/").status_code == 200  # other sessions aren't affected
    assert other_browser(app).get("/login").status_code == 200  # nor are signed-out clients
    clock.advance(61)
    assert client.get("/").status_code == 200


def test_open_job_tabs_can_poll_the_status_api(client: TestClient, clock: FakeClock):
    csrf = sign_in(client)
    job_id = finished_job(client, csrf)
    for _ in range(40):  # 10 tabs polling every 1.5 s for a minute: 400 polls
        for _ in range(10):
            assert client.get(f"/jobs/{job_id}/status").status_code == 200
        clock.advance(1.5)
    assert client.get("/").status_code == 200  # the polls didn't use up the page budget


def test_status_api_has_its_own_limit(client: TestClient, clock: FakeClock):
    csrf = sign_in(client)
    job_id = finished_job(client, csrf)
    clock.advance(61)  # a fresh window
    for _ in range(1200):
        assert client.get(f"/jobs/{job_id}/status").status_code == 200
    response = client.get(f"/jobs/{job_id}/status")
    assert response.status_code == 429 and set(response.json()) == {"error"}
    assert 1 <= int(response.headers["retry-after"]) <= 60
    assert_secure_headers(response)
    assert client.get("/").status_code == 200  # pages have their own budget


def test_rate_limiter_window_and_key_cap():
    clock = FakeClock()
    limiter = RateLimiter(2, 60, clock=clock, max_keys=2)
    assert limiter.hit("a") is None and limiter.hit("a") is None
    assert limiter.hit("a") == 60
    assert limiter.hit("b") is None  # keys are independent
    clock.advance(30)
    assert limiter.hit("a") == 30
    assert limiter.hit("c") == 60  # the key table is full of live keys: refused (fail closed)
    clock.advance(30.5)
    assert limiter.hit("a") is None  # the window slid past the first hits
    assert limiter.hit("c") is None  # expired keys were pruned to make room
    assert retry_after(0.2) == "1" and retry_after(59.1) == "60"
    with pytest.raises(ValueError):
        RateLimiter(0, 60, clock=clock)


def test_rate_limiter_retry_in_counts_nothing():
    clock = FakeClock()
    limiter = RateLimiter(2, 60, clock=clock, max_keys=2)
    assert limiter.retry_in("a") is None
    assert limiter.hit("a") is None
    assert limiter.retry_in("a") is None and limiter.retry_in("a") is None  # peeking counts nothing
    assert limiter.hit("a") is None
    assert limiter.retry_in("a") == 60
    clock.advance(10)
    assert limiter.retry_in("a") == 50
    assert limiter.hit("b") is None
    assert limiter.retry_in("c") == 60  # the key table is full of live keys: refused (fail closed)
    clock.advance(50.5)
    assert limiter.retry_in("a") is None
    assert limiter.retry_in("c") is None  # expired keys were pruned to make room
    assert limiter.hit("c") is None


def test_all_loopback_addresses_are_one_client():
    # Otherwise a local process could get fresh limits from each 127.x.y.z address.
    def key(host: str | None) -> str:
        return client_key({"type": "http", "client": None if host is None else (host, 50000)})

    for host in ("127.0.0.1", "127.8.9.10", "::1", "::ffff:127.0.0.1"):
        assert key(host) == "loopback", host
    assert key("192.168.1.5") == "192.168.1.5"
    assert key("::ffff:10.0.0.1") == "10.0.0.1"
    assert key("testclient") == "testclient"  # the test client's made-up peer name
    assert key(None) == "unknown"


# --------------------------------------------------------------------------- #
# Secrets, logging, and fail-closed setup
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("token", ["", "short", "x" * 31, " " + "x" * 40, "x" * 40 + "\n", "x" * 513, None, b"x" * 40])
def test_create_app_refuses_weak_login_tokens(token):
    with pytest.raises(ValueError):
        create_app(Settings(), login_token=token, mapper_factory=FakeMapper)


def test_create_app_accepts_a_32_character_token():
    assert create_app(Settings(), login_token="x" * 32, mapper_factory=FakeMapper) is not None


def test_secrets_and_personal_data_are_never_logged(client: TestClient, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG)
    wrong = "wrong-token-value-" + "z" * 30
    client.post("/login", data={"csrf_token": csrf_from(client.get("/login").text), "token": wrong})
    csrf = sign_in(client)
    sid = cookie_payload(client.cookies[SESSION_COOKIE])["sid"]
    note = "My SSN is 123-45-6789 and my email is jane@example.com"
    job_id = job_id_of(upload(client, csrf, form=("jane-doe-tax-form.pdf", PDF_BYTES),
                              data=[("jane-doe-record.json", JSON_BYTES)], notes=note))
    wait_for_job(client, job_id)
    client.post(f"/jobs/{job_id}/delete", data={"csrf_token": csrf})
    client.post("/logout", data={"csrf_token": csrf})
    for secret in (TOKEN, wrong, csrf, sid, "123-45-6789", "jane@example.com", "jane-doe", "Jane Q. Public"):
        assert secret not in caplog.text, secret
    assert job_id in caplog.text  # IDs are fine to log


# --------------------------------------------------------------------------- #
# Upload checks and file names
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("field", "filename", "content", "allowed"),
    [
        ("form", "form.pdf", PDF_BYTES, True),
        ("form", "FORM.PDF", PDF_BYTES, True),
        ("form", "form.exe.pdf", PDF_BYTES, True),  # the last extension AND the content decide
        ("form", "form.pdf.exe", PDF_BYTES, False),
        ("form", ".pdf", PDF_BYTES, False),  # a dot file has no extension
        ("form", None, PDF_BYTES, False),
        ("form", "form.pdf", b"", False),
        ("form", "form.pdf", b" " * 1000 + PDF_BYTES, True),  # header within the first 1 KB
        ("form", "form.pdf", b" " * 1020 + PDF_BYTES, False),
        ("form", "form.pdf", PNG_BYTES, False),
        ("references", "old.pdf", PDF_BYTES, True),
        ("references", "old.png", PNG_BYTES, False),
        ("data", "data.json", b'{"a": 1}', True),
        ("data", "data.CSV", "name\nJosé\n".encode(), True),
        ("data", "data.json", b"\xff\xfe{}", False),
        ("data", "data.txt", b"text", False),
        ("docs", "scan.png", PNG_BYTES, True),
        ("docs", "scan.png", JPEG_BYTES, False),
        ("docs", "photo.jpeg", JPEG_BYTES, True),
        ("docs", "photo.jpg", PNG_BYTES, False),
        ("docs", "letter.txt", "Grüße".encode(), True),
        ("docs", "letter.txt", b"a\x00b", False),
        ("docs", "mail.eml", b"From: a@example.com\r\n\r\nHi", True),
        ("docs", "page.html", b"<html>", False),
        ("docs", "image.svg", b"<svg/>", False),
        ("docs", "../../scan.png", PNG_BYTES, True),  # directories are ignored
        ("docs", "scan.png/evil.exe", PNG_BYTES, False),
        ("unknown", "form.pdf", PDF_BYTES, False),
    ],
)
def test_upload_allow_list(field: str, filename: str | None, content: bytes, allowed: bool):
    assert is_allowed_upload(field, filename, content) is allowed


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("form.pdf", "form.filled.pdf"),
        ('my "tax" form.pdf', "my_tax_form.filled.pdf"),
        ("a\r\nSet-Cookie: x=y.pdf", "a_Set-Cookie_x_y.filled.pdf"),
        ("../../etc/passwd", "passwd.filled.pdf"),
        ("C:\\Users\\jane\\w-9.PDF", "w-9.filled.pdf"),
        ("application ***-**-6789.pdf", "application_6789.filled.pdf"),
        ("évian.pdf", "vian.filled.pdf"),
        ("....pdf", "filled.pdf"),
        ("", "filled.pdf"),
        (None, "filled.pdf"),
        ("x" * 200 + ".pdf", "x" * 80 + ".filled.pdf"),
    ],
)
def test_download_names_use_an_allow_list(name: str | None, expected: str):
    assert download_name(name) == expected
    assert re.fullmatch(r"[A-Za-z0-9._-]+", expected)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("form.pdf", "form.pdf"),
        ("../../etc/passwd", "passwd"),
        ("C:\\Users\\jane\\form.pdf", "form.pdf"),
        ("invoice\u202efdp.exe", "invoicefdp.exe"),  # bidi override removed
        ("a\x00b\r\n  c.pdf", "ab c.pdf"),
        (None, ""),
    ],
)
def test_display_names_are_safe_to_show(name: str | None, expected: str):
    assert display_name(name) == expected


def test_long_display_names_keep_their_extension():
    shown = display_name("x" * 300 + ".pdf")
    assert len(shown) == 120 and shown.endswith("….pdf")


# --------------------------------------------------------------------------- #
# Templates and front-end code
# --------------------------------------------------------------------------- #


def test_templates_have_no_inline_code_and_quote_every_attribute():
    templates = sorted(TEMPLATES_DIR.glob("*.html"))
    assert {path.name for path in templates} == {"base.html", "login.html", "index.html", "job.html", "error.html"}
    for path in templates:
        html = path.read_text(encoding="utf-8")
        for attributes, body in re.findall(r"<script\b([^>]*)>(.*?)</script>", html, re.S):
            assert re.search(r'\ssrc="/static/[\w.-]+\.js"', attributes) and not body.strip(), path.name
        assert "<style" not in html, path.name
        for tag in re.findall(r"<[a-zA-Z][^>]*>", html):
            bare = re.sub(r'"[^"]*"', '""', tag)  # drop the quoted values
            assert not re.search(r"\s[\w:-]+=[^\"]", bare), (path.name, tag)  # every value is quoted
            assert "{{" not in bare and "{%" not in bare, (path.name, tag)  # template output only inside quotes
            assert not re.search(r"\s(on[a-z]+|style)=", bare, re.I), (path.name, tag)  # no handlers or styles
        assert "javascript:" not in html.lower(), path.name
        assert not re.search(r"https?://", html), path.name  # nothing loaded from elsewhere
        assert "|safe" not in html and "autoescape" not in html, path.name


def test_templates_autoescape(app):
    env = app.state.web.templates.env
    assert env.autoescape is True
    rendered = env.from_string('<p title="{{ v }}">{{ v }}</p>').render(v='"><script>x</script>')
    assert rendered == '<p title="&#34;&gt;&lt;script&gt;x&lt;/script&gt;">&#34;&gt;&lt;script&gt;x&lt;/script&gt;</p>'


def test_front_end_script_avoids_dangerous_apis():
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    for banned in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "alert(", "confirm(",
                   "prompt(", "eval(", "new Function", "console.", "localStorage", "sessionStorage",
                   "http://", "https://", "import("):
        assert banned not in js, banned
    assert 'credentials: "same-origin"' in js


def test_nothing_drops_the_origin_from_the_apps_own_posts(client: TestClient):
    # Under a no-referrer policy (the header, a <meta>, or rel="noreferrer"), browsers send
    # `Origin: null` with form posts, and the cross-site filter rejects them.
    sources = [*sorted(TEMPLATES_DIR.glob("*.html")), STATIC_DIR / "app.js"]
    for path in sources:
        text = path.read_text(encoding="utf-8").lower()
        for banned in ("no-referrer", "noreferrer", "referrerpolicy"):
            assert banned not in text, (path.name, banned)
    page = client.get("/login")
    assert re.findall(r'<meta name="referrer" content="([^"]*)"', page.text) == ["same-origin"]
    assert page.headers["referrer-policy"] == "same-origin"


def test_styles_and_images_load_nothing_external():
    css = (STATIC_DIR / "app.css").read_text(encoding="utf-8")
    assert "@import" not in css and "url(" not in css and "expression(" not in css
    for name in ("logo.svg", "favicon.svg"):
        svg = (STATIC_DIR / name).read_text(encoding="utf-8")
        assert "<script" not in svg and "foreignObject" not in svg and "href" not in svg, name
        assert not re.search(r"\son[a-z]+=", svg, re.I), name
        assert re.findall(r"https?://[^\"]+", svg) == ["http://www.w3.org/2000/svg"], name


def _eager_imports(node: ast.AST) -> Iterator[str]:
    """Modules imported when `node` runs: skips function bodies and TYPE_CHECKING blocks."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(child, ast.If) and "TYPE_CHECKING" in ast.unparse(child.test):
            continue
        if isinstance(child, ast.Import):
            yield from (alias.name for alias in child.names)
        elif isinstance(child, ast.ImportFrom):
            module = child.module or ""
            yield module
            yield from (f"{module}.{alias.name}" for alias in child.names)
        else:
            yield from _eager_imports(child)


def test_web_package_imports_the_core_lazily():
    # The pipeline and sources modules are imported inside the job runner only, so
    # the web package imports even when they are unavailable.
    for path in sorted(WEB_DIR.glob("*.py")):
        modules = set(_eager_imports(ast.parse(path.read_text(encoding="utf-8"))))
        eager = {m for m in modules if {"pipeline", "sources"} & set(m.split("."))}
        assert not eager, (path.name, eager)
