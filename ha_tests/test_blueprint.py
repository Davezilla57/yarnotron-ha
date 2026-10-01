"""The blueprint is valid, and running it on a real alert reaches its actions."""

from __future__ import annotations

from pathlib import Path

from homeassistant.components.blueprint import models
from homeassistant.components.blueprint.schemas import BLUEPRINT_SCHEMA
from homeassistant.core import HomeAssistant, callback
from homeassistant.setup import async_setup_component
from homeassistant.util import yaml as yaml_util

BLUEPRINT = (
    Path(__file__).resolve().parents[1] / "blueprints/automation/yarnotron/safety_alert_notify.yaml"
)


def test_blueprint_parses_and_has_required_inputs() -> None:
    data = yaml_util.load_yaml(str(BLUEPRINT))
    assert data["blueprint"]["domain"] == "automation"
    assert set(data["blueprint"]["input"]) == {"alert_entity", "alert_types", "notify_action"}


async def test_blueprint_runs_actions_on_a_blocked_alert(
    hass: HomeAssistant, backend, setup
) -> None:
    fake, _ = backend
    await setup()
    calls: list = []

    @callback
    def record(call) -> None:
        calls.append(call.data)

    hass.services.async_register("test", "notify", record)

    blueprint = models.Blueprint(
        yaml_util.load_yaml(str(BLUEPRINT)),
        expected_domain="automation",
        path="yarnotron/x.yaml",
        schema=BLUEPRINT_SCHEMA,
    )
    inputs = models.BlueprintInputs(
        blueprint,
        {
            "use_blueprint": {
                "path": "yarnotron/x.yaml",
                "input": {
                    "alert_entity": "event.yarnotron_safety_alert",
                    "alert_types": ["blocked"],
                    "notify_action": [
                        {
                            "action": "test.notify",
                            "data": {
                                "message": "{{ alert_type }}:{{ event_id }}:{{ categories | join(',') }}"
                            },
                        }
                    ],
                },
            }
        },
    )
    config = inputs.async_substitute()
    assert await async_setup_component(hass, "automation", {"automation": [config]})
    await hass.async_block_till_done()

    entry = hass.config_entries.async_entries("yarnotron")[0]
    # A caution is not in the chosen types; a block is.
    fake.add_event("c1", action="redirect", categories=["scary"])
    await entry.runtime_data.coordinator.async_refresh()
    await hass.async_block_till_done()
    assert calls == []

    fake.add_event("b1", action="block", categories=["violence"])
    await entry.runtime_data.coordinator.async_refresh()
    await hass.async_block_till_done()
    assert calls == [{"message": "blocked:b1:violence"}]
    assert "something scary" not in str(calls)
