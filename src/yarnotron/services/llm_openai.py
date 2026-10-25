"""A model-server client for OpenAI-compatible endpoints (LiteLLM, vLLM, llama.cpp, ...).

It has the same five methods as `OllamaClient` and is a subclass of it, so every
caller keeps working unchanged: the story model, the safety reviewer, interest
summaries and the embedder only ever call `list_models`, `generate`,
`generate_json`, `stream_json` and `embed`.

What differs from Ollama's native API:

* Chat completions instead of `/api/generate`: the prompt becomes one user message.
* `num_predict` becomes `max_tokens`; `think: false` becomes
  `chat_template_kwargs.enable_thinking = false` plus `reasoning_effort = "none"` (see `llm_thinking_hint`).
* Streaming is server-sent events (`data: {...}` lines ending in `[DONE]`).
* Ollama's per-call counters do not exist, so `StreamEvent.stats` is built from the
  `usage` block (when the server sends one) and local clocks: `prompt_eval_duration`
  is the time to the first token, `eval_duration` the time after it.
* The reply is parsed as tolerantly as ever: a JSON fence or prose around the object
  is stripped. `llm_json_mode` additionally asks for `response_format: json_object`.

Failures raise the same exceptions as the Ollama client, so the API routes still turn
"model server down" into a 503 and the safety reviewer still retries unreadable JSON.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from yarnotron.core.config import Settings
from yarnotron.services.incremental_json import JsonStringFieldReader
from yarnotron.services.ollama import (
    OllamaClient,
    OllamaMalformedJsonError,
    OllamaModel,
    OllamaUnavailableError,
    StreamEvent,
    _extract_json_object,
)

# Ollama option name -> OpenAI request field. Anything else is not forwarded.
_OPTION_FIELDS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "seed": "seed",
    "stop": "stop",
    "num_predict": "max_tokens",
}


class OpenAICompatClient(OllamaClient):
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        super().__init__(settings, client)
        if not settings.llm_base_url:
            raise ValueError("YARNOTRON_LLM_BASE_URL is required for the openai backend")
        self._base_url = settings.llm_base_url.rstrip("/")

    # ------------------------------------------------------------------ requests

    def _headers(self) -> dict[str, str]:
        key = self._settings.llm_api_key
        return {"Authorization": f"Bearer {key}"} if key else {}

    def _unavailable(self, exc: BaseException) -> OllamaUnavailableError:
        return OllamaUnavailableError(
            f"The model server is unavailable at {self._base_url}. "
            "Start it or update YARNOTRON_LLM_BASE_URL / YARNOTRON_LLM_API_KEY."
        )

    def _chat_body(
        self,
        prompt: str,
        model: str | None,
        options: dict[str, Any] | None,
        think: bool | None,
        *,
        stream: bool,
        json_mode: bool,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model or self._settings.default_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
        }
        for name, value in (options or {}).items():
            field = _OPTION_FIELDS.get(name)
            if field is not None and value is not None:
                body[field] = value
        if think is not None and self._settings.llm_thinking_hint:
            # Two spellings of "no reasoning", because engines disagree: llama.cpp, vLLM and Qwen
            # templates read chat_template_kwargs; Ollama's OpenAI endpoint reads reasoning_effort
            # (and ignores the template kwarg, so a thinking model spends its whole token budget
            # reasoning). Servers that know neither ignore both.
            body["chat_template_kwargs"] = {"enable_thinking": bool(think)}
            if not think:
                body["reasoning_effort"] = "none"
        if json_mode and self._settings.llm_json_mode:
            body["response_format"] = {"type": "json_object"}
        if stream:
            body["stream_options"] = {"include_usage": True}
        return body

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = await self._ensure_client().request(
                method, f"{self._base_url}{path}", headers=self._headers(), **kwargs
            )
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise self._unavailable(exc) from exc

    # ------------------------------------------------------------------ the five methods

    async def list_models(self) -> list[OllamaModel]:
        payload = await self._request("GET", "/models")
        return [
            OllamaModel(name=str(item["id"]))
            for item in payload.get("data", [])
            if isinstance(item, dict) and item.get("id")
        ]

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        *,
        options: dict[str, Any] | None = None,
        think: bool | None = False,
    ) -> str:
        payload = await self._request(
            "POST",
            "/chat/completions",
            json=self._chat_body(prompt, model, options, think, stream=False, json_mode=False),
        )
        return _reply_text(payload).strip()

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
        payload = await self._request(
            "POST",
            "/chat/completions",
            json=self._chat_body(
                prompt, model, request_options, think, stream=False, json_mode=True
            ),
        )
        return _parse_object(_reply_text(payload).strip())

    async def stream_json(
        self,
        prompt: str,
        model: str | None = None,
        *,
        field: str = "response_text",
        options: dict[str, Any] | None = None,
        think: bool | None = False,
    ) -> AsyncIterator[StreamEvent]:
        request_options = self._base_options()
        if options:
            request_options.update(options)
        request_json = self._chat_body(
            prompt, model, request_options, think, stream=True, json_mode=True
        )

        reader = JsonStringFieldReader(field)
        chunks: list[str] = []
        usage: dict[str, Any] | None = None
        started = time.perf_counter()
        first_token_at: float | None = None
        url = f"{self._base_url}/chat/completions"

        try:
            async with self._ensure_client().stream(
                "POST", url, json=request_json, headers=self._headers()
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if isinstance(chunk.get("usage"), dict):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        piece = ((choice.get("delta") or {}).get("content")) or ""
                        if not piece:
                            continue
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        chunks.append(piece)
                        was_done = reader.done
                        text = reader.feed(piece)
                        if not was_done and reader.done:
                            yield StreamEvent(text=text, field_done=True)
                        elif text:
                            yield StreamEvent(text=text)
        except (httpx.HTTPError, ValueError) as exc:
            raise self._unavailable(exc) from exc

        ended = time.perf_counter()
        stats: dict[str, Any] = {"load_duration": 0}
        if usage:
            for source, target in (
                ("prompt_tokens", "prompt_eval_count"),
                ("completion_tokens", "eval_count"),
            ):
                if source in usage:
                    stats[target] = usage[source]
        if first_token_at is not None:
            stats["prompt_eval_duration"] = int((first_token_at - started) * 1e9)
            stats["eval_duration"] = int((ended - first_token_at) * 1e9)

        yield StreamEvent(payload=_parse_object("".join(chunks).strip()), stats=stats)

    async def embed(self, text: str, model: str) -> list[float]:
        payload = await self._request(
            "POST", "/embeddings", json={"model": model, "input": text}
        )
        data = payload.get("data")
        if isinstance(data, list) and data:
            embedding = data[0].get("embedding") if isinstance(data[0], dict) else None
            if isinstance(embedding, list) and embedding:
                return [float(value) for value in embedding]
        raise OllamaUnavailableError("The model server did not return a usable embedding.")


def _reply_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise OllamaMalformedJsonError("The model server returned no choices.")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    return str(content or "")


def _parse_object(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = _extract_json_object(raw)
    if not isinstance(parsed, dict):
        raise OllamaMalformedJsonError("The model server returned JSON, but it was not an object.")
    return parsed
