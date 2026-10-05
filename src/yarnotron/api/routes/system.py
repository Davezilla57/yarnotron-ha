from fastapi import APIRouter, Depends, Query

from yarnotron.models.status import PrepareShutdownResponse, StorySaved
from yarnotron.services.shutdown import ShutdownCoordinator, get_shutdown
from yarnotron.services.story_store import StoryRepository, get_story_repository

router = APIRouter(prefix="/system", tags=["system"])


@router.post("/prepare-shutdown", response_model=PrepareShutdownResponse)
async def prepare_shutdown(
    timeout: float = Query(default=30.0, ge=0, le=300),
    repository: StoryRepository = Depends(get_story_repository),
    shutdown: ShutdownCoordinator = Depends(get_shutdown),
) -> PrepareShutdownResponse:
    """Get the backend safe to stop, then say what happened.

    Stops taking new turns, waits for running turns and deferred safety reviews,
    ends open story sessions, and folds the database log into the file. Safe to
    call twice. It does not exit the process; the supervisor does that next.
    """
    notes: list[str] = []
    shutdown.begin()
    report = await shutdown.drain(timeout)
    if report.turns_still_running:
        notes.append(f"{report.turns_still_running} turn(s) were still running after {timeout:g}s.")
    if report.background_tasks_cancelled:
        notes.append(
            f"{report.background_tasks_cancelled} background safety/interest task(s) "
            "were cancelled before finishing."
        )

    sessions = await repository.list_sessions()
    open_sessions = [session for session in sessions if session.ended_at is None]
    ended: list[str] = []
    for session in open_sessions:
        await repository.end_session(session.id)
        ended.append(session.id)

    # The story the child was in: the most recently touched among the sessions we ended,
    # or the latest session overall when nothing was open.
    story = None
    reference = (open_sessions or sorted(sessions, key=lambda s: s.started_at))[-1:]
    if reference:
        found = await repository.get(reference[0].story_id)
        if found is not None:
            story = StorySaved(
                story_id=found.id,
                title=found.title,
                scenes=len(found.scenes),
                characters=len(found.characters),
                updated_at=found.updated_at.isoformat(),
            )

    try:
        checkpoint = await repository.checkpoint()
    except Exception as exc:  # noqa: BLE001 - report it, don't fail the shutdown
        checkpoint = f"failed: {type(exc).__name__}: {exc}"
        notes.append(
            f"Database checkpoint {checkpoint}. The data is committed; only the log was not folded in."
        )

    return PrepareShutdownResponse(
        draining=shutdown.draining,
        drained=report.drained,
        waited_seconds=report.waited_seconds,
        turns_in_flight_at_start=report.turns_in_flight_at_start,
        turns_still_running=report.turns_still_running,
        background_tasks_finished=report.background_tasks_finished,
        background_tasks_cancelled=report.background_tasks_cancelled,
        sessions_ended=ended,
        story=story,
        checkpoint=checkpoint,
        notes=notes,
    )


@router.post("/cancel-shutdown")
async def cancel_shutdown(shutdown: ShutdownCoordinator = Depends(get_shutdown)) -> dict[str, bool]:
    """Resume taking turns after a prepare-shutdown that was not followed through."""
    shutdown.cancel()
    return {"draining": shutdown.draining}
