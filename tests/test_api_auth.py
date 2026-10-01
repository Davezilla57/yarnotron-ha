"""Optional bearer-token protection for the API.

Yarnotron used to trust everything that could reach its port. That is fine on a
laptop and wrong on a home network: the parent dashboard can delete stories and
release safety blocks, and the child's other devices are on the same LAN.
Setting YARNOTRON_API_TOKEN closes /v1; leaving it unset changes nothing.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from yarnotron.api.app import create_app
from yarnotron.core.config import get_settings

TOKEN = "s3cret-token-value"


def make_client(monkeypatch, tmp_path, token: str | None) -> TestClient:
    monkeypatch.setenv("YARNOTRON_DATABASE_URL", str(tmp_path / "y.db"))
    monkeypatch.setenv("YARNOTRON_MEDIA_OUTPUT_DIR", str(tmp_path / "media"))
    monkeypatch.setenv("YARNOTRON_PREFERENCES_PATH", str(tmp_path / "prefs.json"))
    if token is None:
        monkeypatch.delenv("YARNOTRON_API_TOKEN", raising=False)
    else:
        monkeypatch.setenv("YARNOTRON_API_TOKEN", token)
    get_settings.cache_clear()
    return TestClient(create_app())


@pytest.fixture(autouse=True)
def _reset_settings():
    yield
    get_settings.cache_clear()


def test_no_token_configured_leaves_the_api_open(monkeypatch, tmp_path):
    with make_client(monkeypatch, tmp_path, None) as client:
        assert client.get("/v1/stories").status_code == 200


def test_blank_token_is_the_same_as_none(monkeypatch, tmp_path):
    # `YARNOTRON_API_TOKEN=` in a .env file must not lock everyone out.
    with make_client(monkeypatch, tmp_path, "") as client:
        assert client.get("/v1/stories").status_code == 200


def test_missing_token_is_rejected(monkeypatch, tmp_path):
    with make_client(monkeypatch, tmp_path, TOKEN) as client:
        response = client.get("/v1/stories")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_wrong_token_is_rejected(monkeypatch, tmp_path):
    with make_client(monkeypatch, tmp_path, TOKEN) as client:
        response = client.get("/v1/stories", headers={"Authorization": "Bearer nope"})

    assert response.status_code == 401


def test_other_schemes_are_rejected(monkeypatch, tmp_path):
    with make_client(monkeypatch, tmp_path, TOKEN) as client:
        response = client.get("/v1/stories", headers={"Authorization": f"Basic {TOKEN}"})

    assert response.status_code == 401


def test_correct_token_is_accepted(monkeypatch, tmp_path):
    with make_client(monkeypatch, tmp_path, TOKEN) as client:
        response = client.get("/v1/stories", headers={"Authorization": f"Bearer {TOKEN}"})

    assert response.status_code == 200


def test_scheme_is_case_insensitive(monkeypatch, tmp_path):
    with make_client(monkeypatch, tmp_path, TOKEN) as client:
        response = client.get("/v1/stories", headers={"Authorization": f"bearer {TOKEN}"})

    assert response.status_code == 200


def test_every_v1_route_family_is_protected(monkeypatch, tmp_path):
    paths = [
        ("GET", "/v1/stories"),
        ("GET", "/v1/sessions"),
        ("GET", "/v1/safety-events"),
        ("GET", "/v1/media"),
        ("GET", "/v1/preferences"),
        ("GET", "/v1/models/ollama"),
        ("POST", "/v1/memory/search"),
        ("POST", "/v1/agent/turn"),
        ("POST", "/v1/agent/turn/stream"),
        ("DELETE", "/v1/stories/story_x"),
        ("POST", "/v1/safety-events/safety_x/release"),
    ]
    with make_client(monkeypatch, tmp_path, TOKEN) as client:
        for method, path in paths:
            response = client.request(method, path, json={})
            assert response.status_code == 401, f"{method} {path} was reachable without a token"


def test_health_and_dashboard_page_stay_open(monkeypatch, tmp_path):
    # Health must answer so a client can tell "down" from "wrong token". The
    # dashboard page itself holds no data; its API calls are what is protected.
    with make_client(monkeypatch, tmp_path, TOKEN) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/dashboard").status_code == 200


def test_dashboard_sends_the_token_with_its_requests(monkeypatch, tmp_path):
    with make_client(monkeypatch, tmp_path, TOKEN) as client:
        page = client.get("/dashboard").text

    assert "Authorization" in page
    assert "sessionStorage" in page
