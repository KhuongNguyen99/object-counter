"""Browser asset caching should revalidate across UI deployments."""

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


STATIC_PATHS = (
    "/",
    "/index.html",
    "/styles.css",
    "/app.js",
    "/reference-review.js",
    "/reference-review.css",
    "/reference-selection.js",
)


@pytest.fixture
def client(tmp_path):
    # Static files and health never initialize the inference engine.
    app = create_app(tmp_path, engine_factory=lambda: None, model_lister=lambda: [])
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.parametrize("path", STATIC_PATHS)
def test_static_assets_revalidate_and_keep_etags(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"
    etag = response.headers["etag"]

    unchanged = client.get(path, headers={"If-None-Match": etag})
    assert unchanged.status_code == 304
    assert unchanged.headers["cache-control"] == "no-cache"
    assert unchanged.headers["etag"] == etag


def test_index_serves_reference_review_ui(client):
    html = client.get("/").text
    assert 'src="/reference-review.js' in html
    assert 'id="reference-inspector"' in html
    assert 'id="candidate-list"' not in html
    assert 'id="reference-selected-list"' in html
    assert 'id="reference-magnifier"' in html
    assert 'id="reference-image-viewport"' in html
    assert 'id="reference-select-areas"' in html
    assert 'id="detection-area-markers"' in html
    assert 'src="/reference-selection.js' in html
    assert 'id="candidate-help"' not in html
    assert 'id="box-list"' not in html


def test_cache_policy_is_limited_to_static_assets(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "cache-control" not in response.headers
