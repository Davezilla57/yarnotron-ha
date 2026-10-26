from __future__ import annotations

from yarnotron.core.config import Settings
from yarnotron.services.ollama import OllamaClient


async def test_generate_json_disables_thinking_and_caps_output() -> None:
    client = OllamaClient(Settings())
    captured: dict[str, object] = {}

    async def fake_request(method: str, path: str, **kwargs: object) -> dict[str, object]:
        captured["method"] = method
        captured["path"] = path
        captured["json"] = kwargs["json"]
        return {"response": '{"ok": true}'}

    client._request = fake_request  # type: ignore[method-assign]

    result = await client.generate_json("Return JSON.")

    assert result == {"ok": True}
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/generate"
    assert captured["json"] == {
        "model": "gemma4:12b-mlx",
        "prompt": "Return JSON.",
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_predict": 512},
        "think": False,
    }


async def test_generate_json_merges_low_latency_options() -> None:
    client = OllamaClient(Settings())
    captured: dict[str, object] = {}

    async def fake_request(method: str, path: str, **kwargs: object) -> dict[str, object]:
        captured["json"] = kwargs["json"]
        return {"response": '{"ok": true}'}

    client._request = fake_request  # type: ignore[method-assign]

    await client.generate_json(
        "Return JSON.",
        model="qwen3.5:2b-mlx",
        options={"num_predict": 160},
        think=False,
    )

    assert captured["json"] == {
        "model": "qwen3.5:2b-mlx",
        "prompt": "Return JSON.",
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_predict": 160},
        "think": False,
    }


def test_a_mangled_json_object_is_a_malformed_reply_not_a_crash():
    import pytest

    from yarnotron.services.ollama import OllamaMalformedJsonError, _extract_json_object

    with pytest.raises(OllamaMalformedJsonError):
        _extract_json_object('{"response_text": "hi" "story_patches": []}')
