"""Give the child more time to pause before Assist decides they have finished speaking.

Home Assistant exposes this per voice satellite as a select entity ("Finished speaking detection") with the options
default, relaxed and aggressive. The integration sets the ones the parent picked, once Home Assistant has started.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


async def async_apply_finished_speaking(hass: HomeAssistant, entity_ids: Iterable[str], mode: str) -> list[str]:
    """Set each select entity to `mode`. Returns the entities that were changed. Never raises for a missing one."""
    changed: list[str] = []
    for entity_id in entity_ids:
        state = hass.states.get(entity_id)
        if state is None:
            _LOGGER.warning("Finished-speaking select %s does not exist (yet); not changed", entity_id)
            continue
        if mode not in (state.attributes.get("options") or []):
            _LOGGER.warning("%s has no option %r; not changed", entity_id, mode)
            continue
        if state.state == mode:
            continue
        try:
            await hass.services.async_call(
                "select", "select_option", {"entity_id": entity_id, "option": mode}, blocking=True
            )
        except Exception:  # noqa: BLE001 - one bad satellite must not stop the integration loading
            _LOGGER.warning("Could not set %s to %s", entity_id, mode, exc_info=True)
            continue
        changed.append(entity_id)
    return changed
