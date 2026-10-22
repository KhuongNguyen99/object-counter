"""The backend serves API routes and permits only configured frontend origins."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import MAX_BODY_BYTES, create_app


FRONTEND = "https://count.example.test"
OTHER_FRONTEND = "http://localhost:3000"
MODELS = [{"id": "yoloe-26s-seg", "available": True, "reference_mode": "shared"}]


def no_engine():
    raise AssertionError("The inference engine should not start for API and CORS checks")


def make_client(tmp_path, origins=""):
    return TestClient(create_app(tmp_path, engine_factory=no_engine,
                                 model_lister=lambda: MODELS, frontend_origins=origins))


def png():
    output = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(output, format="PNG")
    return output.getvalue()


def test_backend_has_api_and_docs_but_no_frontend(tmp_path):
    with make_client(tmp_path) as client:
        assert client.get("/api/health").json() == {"status": "ok"}
        assert client.get("/api/models").json()["models"] == MODELS
        assert client.get("/openapi.json").status_code == 200
        assert client.get("/docs").status_code == 200
        for path in ("/", "/index.html", "/app.js", "/styles.css",
                     "/reference-review.js", "/unrecognized/path"):
            assert client.get(path).status_code == 404
        assert client.get("/reports/reliability.html").status_code == 404


def test_private_report_route_remains_available(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "reliability.html").write_text("<h1>Report</h1>", encoding="utf-8")
    with make_client(tmp_path) as client:
        response = client.get("/reports/reliability.html")
        assert response.status_code == 200
        assert response.text == "<h1>Report</h1>"


@pytest.mark.parametrize("invalid", [
    "*", "https://*.example.test", "ftp://count.example.test",
    "//count.example.test", "count.example.test", "https://user@count.example.test",
    "https://count.example.test/path", "https://count.example.test//",
    "https://count.example.test?query=1", "https://count.example.test#fragment",
    "https://count.example.test:", "https://count.example.test:65536",
    "https://count.example.test:0", "http://bad host.test",
])
def test_invalid_frontend_origins_fail_at_app_creation(tmp_path, invalid):
    with pytest.raises(ValueError, match="FRONTEND_ORIGINS"):
        create_app(tmp_path, engine_factory=no_engine, model_lister=lambda: MODELS,
                   frontend_origins=invalid)


def test_origins_are_loaded_from_environment_at_app_creation(tmp_path, monkeypatch):
    monkeypatch.setenv("FRONTEND_ORIGINS", f" {FRONTEND}/, ,{OTHER_FRONTEND} ")
    with TestClient(create_app(tmp_path, engine_factory=no_engine,
                               model_lister=lambda: MODELS)) as client:
        for origin in (FRONTEND, OTHER_FRONTEND):
            response = client.get("/api/health", headers={"Origin": origin})
            assert response.headers["access-control-allow-origin"] == origin
        denied = client.get("/api/health", headers={"Origin": "https://elsewhere.test"})
        assert "access-control-allow-origin" not in denied.headers


def test_explicit_origin_list_overrides_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("FRONTEND_ORIGINS", OTHER_FRONTEND)
    with TestClient(create_app(tmp_path, engine_factory=no_engine,
                               model_lister=lambda: MODELS,
                               frontend_origins=[f"{FRONTEND}/"])) as client:
        assert client.get("/api/health", headers={"Origin": FRONTEND}).headers[
            "access-control-allow-origin"] == FRONTEND
        assert "access-control-allow-origin" not in client.get(
            "/api/health", headers={"Origin": OTHER_FRONTEND}).headers


@pytest.mark.parametrize("origin", [
    "https://count.example.test.evil.test", "http://count.example.test",
    "https://count.example.test:443", "https://elsewhere.test", "null",
])
def test_preflight_denies_unlisted_origins(tmp_path, origin):
    with make_client(tmp_path, f"{FRONTEND}, {OTHER_FRONTEND}") as client:
        response = client.options("/api/sessions", headers={
            "Origin": origin, "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        })
        assert response.status_code == 400
        assert "access-control-allow-origin" not in response.headers


def test_preflight_allows_only_configured_origin_method_and_header(tmp_path):
    with make_client(tmp_path, f"{FRONTEND}/, {OTHER_FRONTEND}") as client:
        headers = {"Origin": FRONTEND, "Access-Control-Request-Method": "POST",
                   "Access-Control-Request-Headers": "Content-Type"}
        response = client.options("/api/sessions", headers=headers)
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == FRONTEND
        assert response.headers["vary"] == "Origin"
        assert "POST" in response.headers["access-control-allow-methods"]
        assert "GET" in response.headers["access-control-allow-methods"]
        assert "content-type" in response.headers["access-control-allow-headers"].lower()
        assert "access-control-allow-credentials" not in response.headers
        assert client.options("/api/sessions", headers={**headers,
                              "Access-Control-Request-Method": "DELETE"}).status_code == 400
        assert client.options("/api/sessions", headers={**headers,
                              "Access-Control-Request-Headers": "Authorization"}).status_code == 400


def test_actual_api_responses_and_errors_reflect_only_allowed_origins(tmp_path):
    with make_client(tmp_path, FRONTEND) as client:
        for path in ("/api/health", "/api/models", "/api/sessions/missing", "/"):
            response = client.get(path, headers={"Origin": FRONTEND})
            assert response.headers["access-control-allow-origin"] == FRONTEND
            assert response.headers["vary"] == "Origin"
            denied = client.get(path, headers={"Origin": OTHER_FRONTEND})
            assert "access-control-allow-origin" not in denied.headers

        upload = client.post("/api/sessions", headers={"Origin": FRONTEND},
                             data={"model": "yoloe-26s-seg", "reference_boxes": "[[[0,0,4,4]]]"},
                             files={"references": ("ref.png", png(), "image/png")})
        assert upload.status_code == 201, upload.text
        assert upload.headers["access-control-allow-origin"] == FRONTEND
        assert upload.json()["references"][0]["url"].startswith("/api/")
        denied_upload = client.post("/api/sessions", headers={"Origin": OTHER_FRONTEND},
                                    data={"model": "missing"})
        assert denied_upload.status_code == 422
        assert "access-control-allow-origin" not in denied_upload.headers

        session_id = upload.json()["id"]
        review_error = client.post(f"/api/sessions/{session_id}/images/missing/review",
                                   headers={"Origin": FRONTEND},
                                   json={"revision": 0, "boxes": [], "reviewed": True})
        assert review_error.status_code == 404
        assert review_error.headers["access-control-allow-origin"] == FRONTEND
        denied_review = client.post(f"/api/sessions/{session_id}/images/missing/review",
                                    headers={"Origin": OTHER_FRONTEND},
                                    json={"revision": 0, "boxes": [], "reviewed": True})
        assert denied_review.status_code == 404
        assert "access-control-allow-origin" not in denied_review.headers

        csv_response = client.get(f"/api/sessions/{session_id}/export.csv",
                                  headers={"Origin": FRONTEND})
        assert csv_response.status_code == 200
        assert "attachment;" in csv_response.headers["content-disposition"]
        assert csv_response.headers["access-control-expose-headers"] == "Content-Disposition"


def test_body_limit_error_has_cors_and_default_allows_no_cross_origin(tmp_path):
    with make_client(tmp_path, FRONTEND) as client:
        response = client.post("/api/sessions", headers={
            "Origin": FRONTEND, "Content-Length": str(MAX_BODY_BYTES + 1),
        })
        assert response.status_code == 413
        assert response.headers["access-control-allow-origin"] == FRONTEND

    with make_client(tmp_path) as client:
        response = client.get("/api/health", headers={"Origin": FRONTEND})
        assert response.status_code == 200
        assert "access-control-allow-origin" not in response.headers


def test_unexpected_error_has_cors_only_for_allowed_origin(tmp_path):
    app = create_app(tmp_path, engine_factory=no_engine, model_lister=lambda: MODELS,
                     frontend_origins=FRONTEND)

    @app.get("/api/failing-test-route")
    def fail():
        raise RuntimeError("test failure")

    with TestClient(app, raise_server_exceptions=False) as client:
        allowed = client.get("/api/failing-test-route", headers={"Origin": FRONTEND})
        denied = client.get("/api/failing-test-route", headers={"Origin": OTHER_FRONTEND})
        assert allowed.status_code == denied.status_code == 500
        assert allowed.headers["access-control-allow-origin"] == FRONTEND
        assert "access-control-allow-origin" not in denied.headers
