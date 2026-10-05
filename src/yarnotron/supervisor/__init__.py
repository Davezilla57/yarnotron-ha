"""Mac-side supervisor: lets Home Assistant check, start and stop the backend processes.

Standard library only, so it runs under launchd with the system Python and keeps
working when the project's virtualenv is the thing that is broken.
"""
