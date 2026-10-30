# Launch, shut down and check Yarnotron by voice

**Optional.** This is for a setup where Yarnotron's backend runs directly on a Mac that may sleep. It is **not used** when the backend runs as an always-on container (the Mac Studio deployment): leave the supervisor address empty and voice power off. Home Assistant runs on the Proxmox VM. This adds a small
**supervisor** on the Mac so Home Assistant can check, start and stop the Mac's processes.

Each step says where you do it: **[Mac]** is Terminal on the MacBook, **[HA]** is the Home Assistant web page.

## What you can say

| Say | Result |
|---|---|
| "Launch Yarnotron" | Wakes the Mac if configured, checks Ollama, Whisper and the backend, starts what is not answering. |
| "Shut down Yarnotron" | Asks "say yes, shut down". On yes: saves the story, drains running work, stops the processes. |
| "Yarnotron status" | Speaks every Home Assistant and Mac component and its state. |

Problems are spoken in one sentence; the full detail is a Home Assistant notification.

## 1. Install the supervisor  **[Mac]**

1. Open Terminal and go to the Yarnotron-HA folder, on the branch with this feature:
   `cd ~/Yarnotron-HA && git checkout feature/voice-power-control`
2. Create the config the first time: `bash scripts/install_supervisor_launchd.sh`
   It copies an example to `~/.yarnotron-supervisor/supervisor.toml` and stops.
3. Edit that file (`open -e ~/.yarnotron-supervisor/supervisor.toml`):
   - `token`: make up a long random string (`openssl rand -hex 24`). Keep it for step 2.
   - Replace `/Users/YOU/` with your home folder in every path, and check `which ollama` and
     `which uv` match the paths in `start`.
   - `bearer_token` under `[service.pre_stop]`: the same value as `YARNOTRON_API_TOKEN` in your
     backend `.env`. Delete the line if you do not use one.
4. Run `bash scripts/install_supervisor_launchd.sh` again. It prints "Supervisor is running".
5. Check it: `curl http://127.0.0.1:8990/v1/health` shows `{"ok": true ...}`.

If port 8990 does not answer, read `~/.yarnotron-supervisor/logs/supervisor.log`.

## 2. Tell Home Assistant  **[HA]**

1. Settings > Devices & services > Yarnotron > Configure.
2. Fill in **Mac supervisor address** (for example `192.168.1.20:8990`) and **Mac supervisor token**
   (the `token` from step 1). Optionally the Mac's **MAC address** (System Settings > Network >
   your connection > Details > Hardware) so "launch" can wake it.
3. Submit. A wrong token is rejected here. An unreachable Mac is accepted (it may be asleep).

You now have Launch and Shut down buttons, Mac services and Mac supervisor sensors, and the voice phrases.

## 3. Keep the Mac available  **[Mac]**

- The LaunchAgent starts at login and runs under `caffeinate -i`, which stops idle sleep but not lid-close sleep.
- System Settings > Battery > Options: enable preventing automatic sleeping on power adapter.
- A sleeping MacBook cannot run anything. Wake-on-LAN over Wi-Fi is unreliable; Ethernet on power works best.

## Turning off voice power for children  **[HA]**

In the options, switch off **Allow launching and shutting down by voice**. "Yarnotron status" still works;
use the buttons or services to launch and shut down.

## Troubleshooting

| You hear | Cause | Fix |
|---|---|---|
| "The Mac is on, but the Yarnotron supervisor isn't running." | Connection refused | `launchctl kickstart -k gui/$(id -u)/com.yarnotron.supervisor` **[Mac]**; the agent only runs while you are logged in. |
| "The Mac isn't answering…" | Asleep, off, or off the network | Wake it; set the MAC address; keep it awake on power. |
| "…rejected my token" | Tokens differ | Make **[HA]** options and `supervisor.toml` match. |
| "X started and then quit straight away" | Process crashed | The notification shows the last log lines; run the command in Terminal. |
| "X started but isn't answering after N seconds" | Still loading, or hung | Raise `start_timeout`; try again. |
| "Something else is using X's port" | Unknown program on the port | The supervisor never kills a process it does not recognise (`process_match`). Stop it yourself. |
| "…refused the save-and-drain request" | Backend failed to save | Nothing was stopped. Fix it, or call `yarnotron.shutdown` with `force` (unfinished safety reviews are lost). |
| "…is still answering after I tried to stop it" | Something restarts it (Ollama's menu-bar app, another launchd job) | Quit that app, or set `stop_policy = "leave"` for the service. |

## Not verified

The supervisor's process handling is tested on Linux with real throwaway processes. It has not run on a Mac:
launchd, `lsof` output, `uv`, Ollama's app behaviour and Wake-on-LAN are untested.
