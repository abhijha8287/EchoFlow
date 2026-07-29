"""Connection-scoped WS orchestration — mirrors design doc
`### Cancel scope boundary` (T3).

`ConnectionManager` owns exactly two kinds of task per live session:

- the **connection-scoped receive loop** (`receive_loop()`), which runs for
  the lifetime of the WebSocket, and
- at most one **utterance-scoped pipeline task**, created fresh per
  `utterance_id` via `asyncio.create_task()`.

These are always independent `asyncio.Task`s. The receive loop never awaits
the utterance task and the two are never entered into the same
`asyncio.TaskGroup`/cancel scope. `interrupt()` (barge-in) cancels only the
utterance task -- the receive loop is untouched and keeps accepting the next
mic frame immediately. If the two were nested in one scope, cancelling the
utterance task on barge-in would propagate and tear down the WebSocket
connection itself, kicking the user off on every interrupt.

`Transport` is a minimal receive-side seam over the real WS connection (e.g.
`starlette.websockets.WebSocket`) so `receive_loop()` can be exercised with a
fake queue in tests instead of a live socket.

`on_frame` (added for T7) is an optional hook invoked for every frame,
before the built-in MIC_PCM/CONTROL_INTERRUPT/UTTERANCE_READY dispatch below
-- it lets a caller (e.g. `voice.session_runtime.SessionRuntime`) observe
every frame and drive FSM/VAD/session logic without needing to fork this
class's own dispatch. `spawn_utterance()` is the public counterpart, letting
that same caller start an utterance-scoped task from outside the frame
dispatch (e.g. on a VAD endpoint detected while handling a mic frame),
without breaking the independent-task guarantee.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

MIC_PCM = "mic_pcm"
UTTERANCE_READY = "utterance_ready"
CONTROL_INTERRUPT = "control_interrupt"
TEXT = "text"


class ConnectionClosed(Exception):
    """Raised by a `Transport.receive()` to signal the connection has ended."""


@dataclass(frozen=True, slots=True)
class InboundFrame:
    kind: str
    data: Any = None


class Transport(Protocol):
    async def receive(self) -> InboundFrame: ...


StartUtterance = Callable[[InboundFrame], Awaitable[None]]
MicFrameHook = Callable[[InboundFrame], None]
FrameHook = Callable[[InboundFrame], None]


class ConnectionManager:
    """One instance per live WebSocket connection."""

    def __init__(
        self,
        transport: Transport,
        start_utterance: StartUtterance,
        on_mic_frame: MicFrameHook | None = None,
        on_frame: FrameHook | None = None,
    ) -> None:
        self._transport = transport
        self._start_utterance = start_utterance
        self._on_mic_frame = on_mic_frame
        self._on_frame = on_frame
        self._utterance_task: asyncio.Task[None] | None = None

    @property
    def utterance_task(self) -> asyncio.Task[None] | None:
        return self._utterance_task

    async def receive_loop(self) -> None:
        """The connection-scoped loop. Runs until the transport closes.

        Deliberately a plain `while True` + `await` -- not a TaskGroup, and
        never awaits the utterance task -- so cancelling the utterance task
        (see `interrupt()`) can never unwind this loop.
        """
        try:
            while True:
                frame = await self._transport.receive()
                self._handle_frame(frame)
        except ConnectionClosed:
            self._cancel_current_utterance()

    def interrupt(self) -> None:
        """Barge-in: cancel only the in-flight utterance task, if any."""
        self._cancel_current_utterance()

    def spawn_utterance(self, frame: InboundFrame) -> asyncio.Task[None]:
        """Public entry point for spawning an utterance-scoped task from
        outside the receive loop's own frame dispatch -- e.g. a caller that
        detected a VAD endpoint while handling a mic frame via `on_frame`.
        Same independent-task guarantee as the `UTTERANCE_READY` frame path
        below (see module docstring)."""
        self._spawn_utterance_task(frame)
        assert self._utterance_task is not None
        return self._utterance_task

    def _handle_frame(self, frame: InboundFrame) -> None:
        if self._on_frame is not None:
            self._on_frame(frame)
        if frame.kind == MIC_PCM:
            if self._on_mic_frame is not None:
                self._on_mic_frame(frame)
        elif frame.kind == CONTROL_INTERRUPT:
            self.interrupt()
        elif frame.kind == UTTERANCE_READY:
            self._spawn_utterance_task(frame)

    def _spawn_utterance_task(self, frame: InboundFrame) -> None:
        # Cancel scope boundary: create_task() gives this its own independent
        # Task, never a `async with asyncio.TaskGroup()` shared with
        # receive_loop() above -- that would let cancellation of this task
        # propagate into the loop that owns the connection.
        self._utterance_task = asyncio.create_task(self._start_utterance(frame))

    def _cancel_current_utterance(self) -> None:
        task = self._utterance_task
        if task is not None and not task.done():
            task.cancel()
