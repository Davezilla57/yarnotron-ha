import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import Depends, Request
from pydantic import BaseModel

from yarnotron.core.config import Settings, get_settings
from yarnotron.services.incremental_json import JsonStringFieldReader


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama service cannot be reached."""


class OllamaMalformedJsonError(OllamaUnavailableError):
    """The model answered, but the answer was not usable JSON.

    Split out because the two want opposite handling in the safety reviewer: a
    model emitting garbage on one garbled transcript is not evidence of danger
    and should not cost the child anything, whereas Ollama being down is a real
    failure the caller must see. Subclassing keeps every existing
    `except OllamaUnavailableError` -- the routes that turn this into a 503 --
    working unchanged.
    """


@dataclass
class StreamEvent:
    """A piece of the streamed field, or the finished object at the end.

    `field_done` marks the event where the field's closing quote was reached.
    That happens well before the rest of the object finishes generating, and it
    is what lets the last sentence be spoken immediately instead of waiting for
    the story patches -- a gap of about three seconds otherwise.
    """

    text: str | None = None
    payload: dict[str, Any] | None = None
    field_done: bool = False
    # Ollama's per-call counters, present on the final event. prompt_eval is the
    # cost of reading the prompt before a single token is produced, so it is a
    # floor on time to first audio that streaming cannot reduce.
    stats: dict[str, Any] | None = None


class OllamaModel(BaseModel):
    name: str
    modified_at: str | None = None
    size: int | None = None


class OllamaClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self._settings = settings
        self._client = client
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def list_models(self) -> list[OllamaModel]:
        payload = await self._request("GET", "/api/tags")
        return [OllamaModel.model_validate(item) for item in payload.get("models", [])]

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        *,
        options: dict[str, Any] | None = None,
        think: bool | None = False,
    ) -> str:
        request_json: dict[str, Any] = {
            "model": model or self._settings.default_model,
            "prompt": prompt,
            "stream": False,
        }
        if options:
            request_json["options"] = options
        if think is not None:
            request_json["think"] = think

        payload = await self._request(
            "POST",
            "/api/generate",
            json=request_json,
        )
        return str(payload.get("response", "")).strip()

    async def generate_json(
        self,
        prompt: str,
        model: str | None = None,
        *,
        options: dict[str, Any] | None = None,
        think: bool | None = False,
    ) -> dict[str, Any]:
        request_options = self._base_options()
        if options:
            request_options.update(options)
        request_json: dict[str, Any] = {
            "model": model or self._settings.default_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": request_options,
        }
        if think is not None:
            request_json["think"] = think

        payload = await self._request(
            "POST",
            "/api/generate",
            json=request_json,
        )
        raw_response = str(payload.get("response", "")).strip()
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            parsed = _extract_json_object(raw_response)
        if not isinstance(parsed, dict):
            raise OllamaMalformedJsonError("Ollama returned JSON, but it was not an object.")
        return parsed

    async def stream_json(
        self,
        prompt: str,
        model: str | None = None,
        *,
        field: str = "response_text",
        options: dict[str, Any] | None = None,
        think: bool | None = False,
    ) -> AsyncIterator[StreamEvent]:
        """Generate a JSON object, emitting one field's text as it is written.

        Yields `StreamEvent(text=...)` as `field` becomes readable, then exactly
        one `StreamEvent(payload=...)` carrying the parsed object. The story
        prompt puts `response_text` first so the reply can be spoken long before
        the patches finish generating.
        """
        request_options = self._base_options()
        if options:
            request_options.update(options)
        request_json: dict[str, Any] = {
            "model": model or self._settings.default_model,
            "prompt": prompt,
            "stream": True,
            "format": "json",
            "options": request_options,
        }
        if think is not None:
            request_json["think"] = think

        reader = JsonStringFieldReader(field)
        chunks: list[str] = []
        stats: dict[str, Any] = {}
        url = f"{self._settings.ollama_base_url.rstrip('/')}/api/generate"

        try:
            async with self._ensure_client().stream("POST", url, json=request_json) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    piece = str(chunk.get("response", ""))
                    if piece:
                        chunks.append(piece)
                        was_done = reader.done
                        text = reader.feed(piece)
                        if not was_done and reader.done:
                            yield StreamEvent(text=text, field_done=True)
                        elif text:
                            yield StreamEvent(text=text)
                    if chunk.get("done"):
                        stats = {
                            k: chunk[k]
                            for k in (
                                "prompt_eval_count",
                                "prompt_eval_duration",
                                "eval_count",
                                "eval_duration",
                                "load_duration",
                            )
                            if k in chunk
                        }
                        break
        except (httpx.HTTPError, ValueError) as exc:
            raise OllamaUnavailableError(
                f"Ollama is unavailable at {self._settings.ollama_base_url}. "
                "Start Ollama or update YARNOTRON_OLLAMA_BASE_URL."
            ) from exc

        combined = "".join(chunks).strip()
        try:
            parsed = json.loads(combined)
        except json.JSONDecodeError:
            # Every gemma draft arrives inside a ```json fence, so this is the
            # normal path for the story model rather than an edge case.
            parsed = _extract_json_object(combined)
        if not isinstance(parsed, dict):
            raise OllamaMalformedJsonError("Ollama returned JSON, but it was not an object.")
        yield StreamEvent(payload=parsed, stats=stats)

    async def embed(self, text: str, model: str) -> list[float]:
        payload = await self._request(
            "POST",
            "/api/embed",
            json={
                "model": model,
                "input": text,
            },
        )
        embeddings = payload.get("embeddings")
        if isinstance(embeddings, list) and embeddings:
            first_embedding = embeddings[0]
            if isinstance(first_embedding, list):
                return [float(value) for value in first_embedding]

        embedding = payload.get("embedding")
        if isinstance(embedding, list):
            return [float(value) for value in embedding]

        raise OllamaUnavailableError("Ollama did not return a usable embedding.")

    def _base_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": 0, "num_predict": 512}
        if self._settings.model_seed is not None:
            options["seed"] = self._settings.model_seed
        return options

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._settings.request_timeout_seconds)
            )
        return self._client

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        url = f"{self._settings.ollama_base_url.rstrip('/')}{path}"
        try:
            response = await self._ensure_client().request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise OllamaUnavailableError(
                f"Ollama is unavailable at {self._settings.ollama_base_url}. "
                "Start Ollama or update YARNOTRON_OLLAMA_BASE_URL."
            ) from exc


def get_ollama_client(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> OllamaClient:
    client = getattr(request.app.state, "ollama_client", None)
    if client is not None:
        return client
    return OllamaClient(settings)


def _extract_json_object(raw_response: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", raw_response, flags=re.DOTALL)
    if match is None:
        raise OllamaMalformedJsonError("Ollama did not return parseable JSON.")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise OllamaMalformedJsonError("Ollama returned JSON, but it was not an object.")
    return parsed
