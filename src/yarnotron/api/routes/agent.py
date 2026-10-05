import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from yarnotron.agent.turn_processor import TurnProcessor, get_turn_processor
from yarnotron.models.agent import AgentTurnRequest, AgentTurnResponse
from yarnotron.services.ollama import OllamaUnavailableError
from yarnotron.services.shutdown import get_shutdown, reject_while_draining

router = APIRouter(prefix="/agent", tags=["agent"], dependencies=[Depends(reject_while_draining)])


@router.post("/turn", response_model=AgentTurnResponse)
async def process_turn(
    request: AgentTurnRequest,
    processor: TurnProcessor = Depends(get_turn_processor),
) -> AgentTurnResponse:
    try:
        with get_shutdown().turn():
            return await processor.process(request)
    except OllamaUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Story not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/turn/stream")
async def process_turn_streaming(
    request: AgentTurnRequest,
    processor: TurnProcessor = Depends(get_turn_processor),
) -> StreamingResponse:
    """Same turn, delivered as newline-delimited JSON as it happens.

    Emits `{"sentence": "..."}` for each safety-gated sentence, then a single
    `{"turn": {...}}` carrying the full AgentTurnResponse, or `{"error": "..."}`.
    The buffered endpoint above stays the contract for the dashboard and tests.
    """
    return StreamingResponse(_turn_events(processor, request), media_type="application/x-ndjson")


async def _turn_events(
    processor: TurnProcessor,
    request: AgentTurnRequest,
) -> AsyncIterator[bytes]:
    # A queue decouples the processor's callback from this generator, so a
    # sentence reaches the client the moment it is gated rather than when the
    # turn finishes.
    queue: asyncio.Queue[dict | None] = asyncio.Queue()

    async def on_sentence(sentence: str) -> None:
        await queue.put({"sentence": sentence})

    async def run() -> None:
        try:
            with get_shutdown().turn():
                response = await processor.process(request, on_sentence=on_sentence)
            await queue.put({"turn": response.model_dump(mode="json")})
        except OllamaUnavailableError as exc:
            await queue.put({"error": str(exc), "status": 503})
        except KeyError:
            await queue.put({"error": "Story not found", "status": 404})
        except ValueError as exc:
            await queue.put({"error": str(exc), "status": 400})
        except Exception as exc:  # never leave the client hanging on the queue
            await queue.put({"error": f"{type(exc).__name__}: {exc}", "status": 500})
        finally:
            await queue.put(None)

    task = asyncio.create_task(run())
    try:
        while True:
            event = await queue.get()
            if event is None:
                break
            yield (json.dumps(event) + "\n").encode()
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
