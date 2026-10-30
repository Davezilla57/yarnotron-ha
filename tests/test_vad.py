"""The end-of-speech setting, against a stand-in for Home Assistant (the module needs no Home Assistant import)."""

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

_SPEC = importlib.util.spec_from_file_location(
    "yarnotron_vad", Path(__file__).parents[1] / "custom_components" / "yarnotron" / "vad.py"
)
vad = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(vad)


class FakeHass:
    def __init__(self, states, fail_for=()):
        self._states = states
        self.calls = []
        self._fail_for = set(fail_for)
        self.states = SimpleNamespace(get=lambda entity_id: self._states.get(entity_id))
        self.services = SimpleNamespace(async_call=self._call)

    async def _call(self, domain, service, data, blocking=False):
        if data["entity_id"] in self._fail_for:
            raise RuntimeError("boom")
        self.calls.append((domain, service, data))


def state(value, options=("default", "relaxed", "aggressive")):
    return SimpleNamespace(state=value, attributes={"options": list(options)})


def run(coro):
    return asyncio.run(coro)


def test_sets_only_the_satellites_that_are_not_already_relaxed():
    hass = FakeHass({"select.a": state("default"), "select.b": state("relaxed")})
    changed = run(vad.async_apply_finished_speaking(hass, ["select.a", "select.b"], "relaxed"))
    assert changed == ["select.a"]
    assert hass.calls == [("select", "select_option", {"entity_id": "select.a", "option": "relaxed"})]


def test_missing_entity_wrong_options_and_a_failing_call_do_not_stop_the_rest():
    hass = FakeHass(
        {"select.ok": state("default"), "select.odd": state("x", options=("a", "b")), "select.bad": state("default")},
        fail_for=["select.bad"],
    )
    changed = run(vad.async_apply_finished_speaking(hass, ["select.gone", "select.odd", "select.bad", "select.ok"], "relaxed"))
    assert changed == ["select.ok"]
