# Mac supervisor issue codes

| Code | Plain meaning | What usually fixes it |
|---|---|---|
| command_not_found | The program to start was not found (launchd has a short PATH) | Use the full path in the supervisor's `start` |
| spawn_failed | The system refused to start it | Check the working directory exists and the program is executable |
| exited_early | Started, then quit at once; the log tail is included | Run the command in Terminal to see the real error |
| start_timeout | Running but never answered its health check | Still loading a model? Wait and retry, or raise `start_timeout` |
| port_in_use | Something else holds the port and is not answering | The supervisor will not kill a process it does not recognise. Stop it by hand |
| dependency_failed | Not tried because something it needs failed | Fix the dependency first (usually Ollama), then launch again |
| prepare_failed | The backend would not save and drain, so it was not stopped | Check the backend; a forced shutdown risks losing unfinished safety reviews |
| not_drained | The backend was still busy | Wait for the story turn to finish and ask again |
| kept_running | Left up because something that uses it could not be stopped | Resolve the earlier error first |
| no_pid | It answers but its process was not found | It may be on another machine, or `lsof` is missing |
| still_running | Still answering after being stopped | Something restarts it (Ollama menu-bar app, a launchd job) |

## Home Assistant side

Piper or Whisper "unavailable" means Home Assistant cannot reach that Wyoming server: restart the Piper
add-on, or launch Yarnotron so the Mac starts Whisper. "Backend link: down" means Home Assistant's poll
of the backend failed; launch Yarnotron. A locked story session is a safety feature, not a fault.
