"""Constants for the Yarnotron integration."""

from __future__ import annotations

from datetime import timedelta
from typing import Final

DOMAIN: Final = "yarnotron"

# Config entry data
CONF_API_TOKEN: Final = "api_token"
CONF_CHILD_ID: Final = "child_id"

# Options
CONF_SCAN_INTERVAL: Final = "scan_interval"
CONF_IDLE_MINUTES: Final = "idle_minutes"
CONF_KEEP_LISTENING: Final = "keep_listening"
CONF_INCLUDE_TRANSCRIPTS: Final = "include_transcripts"
CONF_SUPERVISOR_URL: Final = "supervisor_url"
CONF_SUPERVISOR_TOKEN: Final = "supervisor_token"
CONF_MAC_ADDRESS: Final = "mac_address"
CONF_VOICE_POWER: Final = "voice_power"

DEFAULT_CHILD_ID: Final = "local_child"
DEFAULT_SCAN_INTERVAL: Final = 30  # seconds
DEFAULT_IDLE_MINUTES: Final = 30
DEFAULT_KEEP_LISTENING: Final = True
DEFAULT_INCLUDE_TRANSCRIPTS: Final = False
DEFAULT_VOICE_POWER: Final = True

# Launch and shutdown can take minutes (a model loading, a story turn finishing).
# A voice reply waits this long, then hands over to the notification.
VOICE_POWER_WAIT: Final = 50
# How long a spoken "shut down" waits for "yes" before it is forgotten.
SHUTDOWN_CONFIRM_SECONDS: Final = 60
# Time to let a sleeping Mac wake and the supervisor come up after a magic packet.
WAKE_WAIT_SECONDS: Final = 90
WAKE_POLL_SECONDS: Final = 5
LAUNCH_JOB_TIMEOUT: Final = 420
SHUTDOWN_JOB_TIMEOUT: Final = 180
JOB_POLL_SECONDS: Final = 1.5

EVENT_POWER_FINISHED: Final = "yarnotron_power_finished"

MIN_SCAN_INTERVAL: Final = 10
MAX_SCAN_INTERVAL: Final = 600

# How long to wait on the model between streamed lines. A long story reply can
# pause for a while on a cold model, so this is generous; it is a stall limit,
# not a limit on the whole turn.
STREAM_READ_TIMEOUT: Final = 180
REQUEST_TIMEOUT: Final = 15

# Safety categories the backend puts on a turn that ended the session. Once the
# session is over, a grown-up must start the next one; see session.py.
CATEGORY_STRIKE_LIMIT: Final = "strike_limit_reached"
CATEGORY_SESSION_ENDED: Final = "session_ended"
LOCKING_CATEGORIES: Final = frozenset({CATEGORY_STRIKE_LIMIT, CATEGORY_SESSION_ENDED})

# How many recent safety events each poll fetches. Alerts older than this
# window that arrived while Home Assistant was down are not replayed.
SAFETY_EVENT_WINDOW: Final = 25

ALERT_CAUTION: Final = "caution"
ALERT_BLOCKED: Final = "blocked"
ALERT_SESSION_ENDED: Final = "session_ended"
ALERT_TYPES: Final = [ALERT_CAUTION, ALERT_BLOCKED, ALERT_SESSION_ENDED]

# One short follow-up poll after a turn that raised a safety flag: the backend
# reviews what the child said a moment after replying, so the event may not
# exist yet at the first refresh.
DEFERRED_REVIEW_DELAY: Final = timedelta(seconds=6)

STORAGE_VERSION: Final = 1

# Spoken replies that never touch the backend.
HELP_TEXT: Final = (
    "You can say things like: add a new character, change the castle to a "
    "treehouse, read me a recap, undo the last change, repeat that, or stop listening."
)
SLEEP_TEXT: Final = "Okay. I'll be here whenever you want to keep writing."
STOP_TEXT: Final = "Okay, ending this Yarnotron session."
NOTHING_TO_REPEAT_TEXT: Final = "I don't have anything to repeat yet."
STORYTIME_OFF_TEXT: Final = (
    "Storytime is resting right now. Ask a grown-up if you'd like to write something."
)
LOCKED_TEXT: Final = (
    "We stopped this session earlier. Ask a grown-up to start a new one "
    "with you and your story will be waiting."
)
EMPTY_REPLY_TEXT: Final = "Hmm, I didn't catch that. Could you say it again?"
BACKEND_ERROR_TEXT: Final = (
    "I'm having trouble thinking right now. Please give me a moment, then try again."
)

POWER_DISABLED_TEXT: Final = (
    "Starting and stopping Yarnotron by voice is turned off. "
    "A grown-up can use the Yarnotron buttons in Home Assistant."
)
POWERED_DOWN_TEXT: Final = (
    "Yarnotron is shut down right now. A grown-up can say, launch Yarnotron, to start it again."
)
SHUTDOWN_CANCELLED_TEXT: Final = "Okay, I'll leave Yarnotron running."
SHUTDOWN_CONFIRM_TEXT: Final = (
    "Shut down Yarnotron? I'll save the story, then turn off the story engine and the "
    "speech services on the Mac. Say yes, shut down, to confirm."
)
