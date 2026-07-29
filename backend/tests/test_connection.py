"""Cancel scope boundary integration test (T3) — see design doc
`### Cancel scope boundary` and the Testing section: "This cancellation test
must additionally assert the connection-scoped WS receive loop is still
running and accepting mic audio after the utterance task is cancelled."

No pytest-asyncio dependency is added for this: each async test body is
driven via `asyncio.run()` from a plain sync test function.
"""

from __future__ import annotations

import asyncio

import pytest

from websocket.connection import (
    CONTROL_INTERRUPT,
    MIC_PCM,
    UTTERANCE_READY,
    ConnectionClosed,
    ConnectionManager,
    InboundFrame,
)


class QueueTransport:
    """Fake `Transport` driven by a test via an asyncio.Queue, standing in
    for a live `starlette.websockets.WebSocket`."""

    def __init__(self) -> None:
        self._queue: "asyncio.Queue[InboundFrame | None]" = asyncio.Queue()

    def send(self, frame: InboundFrame) -> None:
        self._queue.put_nowait(frame)

    def close(self) -> None:
        self._queue.put_nowait(None)

    async def receive(self) -> InboundFrame:
        frame = await self._queue.get()
        if frame is None:
            raise ConnectionClosed()
        return frame


async def _hang_forever(frame: InboundFrame) -> None:
    """Stand-in utterance pipeline that only ends via cancellation, same as
    a real in-flight STT/LLM/TTS call from `run_utterance_pipeline` would."""
    await asyncio.Event().wait()


def test_barge_in_cancels_utterance_without_killing_receive_loop():
    asyncio.run(_barge_in_cancels_utterance_without_killing_receive_loop())


async def _barge_in_cancels_utterance_without_killing_receive_loop():
    transport = QueueTransport()
    received_mic_frames: list[InboundFrame] = []

    manager = ConnectionManager(
        transport=transport,
        start_utterance=_hang_forever,
        on_mic_frame=received_mic_frames.append,
    )
    loop_task = asyncio.create_task(manager.receive_loop())

    # VAD endpoint reached -> connection manager spawns an utterance-scoped
    # task, independent of loop_task above.
    transport.send(InboundFrame(UTTERANCE_READY))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    utterance_task = manager.utterance_task
    assert utterance_task is not None
    assert not utterance_task.done()

    # Barge-in, delivered the way a real client would: a control frame
    # through the WS, not a direct method call on the manager.
    transport.send(InboundFrame(CONTROL_INTERRUPT))

    with pytest.raises(asyncio.CancelledError):
        await utterance_task

    # The critical assertion (design doc, Testing section): the
    # connection-scoped receive loop must still be alive, and must still
    # accept + process the next mic frame, after the utterance task above
    # was cancelled.
    assert not loop_task.done()
    transport.send(InboundFrame(MIC_PCM, data=b"\x00\x01"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert len(received_mic_frames) == 1
    assert received_mic_frames[0].data == b"\x00\x01"
    assert not loop_task.done()

    transport.close()
    await loop_task  # receive_loop must exit cleanly on ConnectionClosed


def test_receive_loop_exits_cleanly_on_connection_closed_with_no_utterance():
    asyncio.run(_receive_loop_exits_cleanly_on_connection_closed_with_no_utterance())


async def _receive_loop_exits_cleanly_on_connection_closed_with_no_utterance():
    transport = QueueTransport()
    manager = ConnectionManager(transport=transport, start_utterance=_hang_forever)
    loop_task = asyncio.create_task(manager.receive_loop())

    transport.close()
    await loop_task  # must return, not raise

    assert loop_task.done()
    assert manager.utterance_task is None


def test_second_utterance_replaces_task_reference_independently_of_loop():
    asyncio.run(_second_utterance_replaces_task_reference_independently_of_loop())


async def _second_utterance_replaces_task_reference_independently_of_loop():
    """A completed (or cancelled) utterance must not block the next one --
    the receive loop can spawn a fresh utterance-scoped task at any time."""
    transport = QueueTransport()
    manager = ConnectionManager(transport=transport, start_utterance=_hang_forever)
    loop_task = asyncio.create_task(manager.receive_loop())

    transport.send(InboundFrame(UTTERANCE_READY))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    first_task = manager.utterance_task
    assert first_task is not None and not first_task.done()

    manager.interrupt()
    with pytest.raises(asyncio.CancelledError):
        await first_task

    transport.send(InboundFrame(UTTERANCE_READY))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    second_task = manager.utterance_task
    assert second_task is not None
    assert second_task is not first_task
    assert not second_task.done()

    transport.close()
    await loop_task  # receive_loop cancels the still-pending second_task on close
    with pytest.raises(asyncio.CancelledError):
        await second_task
