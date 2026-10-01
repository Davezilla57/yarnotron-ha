from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator


class BackendClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8989") -> None:
        self.base_url = base_url.rstrip("/")

    def create_session(
        self,
        child_id: str,
        story_id: str | None,
        device_id: str,
    ) -> dict:
        return self._post(
            "/v1/sessions",
            {
                "child_id": child_id,
                "story_id": story_id,
                "device_id": device_id,
            },
        )

    def end_session(self, session_id: str) -> dict:
        return self._post(f"/v1/sessions/{session_id}/end", {})

    def agent_turn(
        self,
        child_id: str,
        story_id: str | None,
        session_id: str,
        device_id: str,
        transcript: str,
    ) -> dict:
        return self._post(
            "/v1/agent/turn",
            {
                "child_id": child_id,
                "story_id": story_id,
                "session_id": session_id,
                "device_id": device_id,
                "transcript": transcript,
            },
        )

    def agent_turn_stream(
        self,
        child_id: str,
        story_id: str | None,
        session_id: str,
        device_id: str,
        transcript: str,
    ) -> Iterator[dict]:
        """Yield each event of a turn as it happens.

        Events are `{"sentence": ...}` as the reply is written, then one
        `{"turn": ...}` with the full response, or `{"error": ..., "status": ...}`.
        Speaking a sentence blocks, which naturally paces reading the rest.
        """
        yield from self._post_stream(
            "/v1/agent/turn/stream",
            {
                "child_id": child_id,
                "story_id": story_id,
                "session_id": session_id,
                "device_id": device_id,
                "transcript": transcript,
            },
        )

    def _post_stream(self, path: str, payload: dict) -> Iterator[dict]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8")
            raise RuntimeError(f"{path} failed ({exc.code}): {detail}") from exc

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8")
            try:
                parsed_detail = json.loads(detail).get("detail", detail)
            except json.JSONDecodeError:
                parsed_detail = detail
            raise RuntimeError(f"{path} failed ({exc.code}): {parsed_detail}") from exc
