"""Metrics persistence — append-only metrics-runs.jsonl, mirrors design doc
`### Metrics persistence` (T5).

Every `metrics.stage` event the server emits is also appended here as one
JSON line, tagged with `provider`, `role`, `stage`, `duration_ms`,
`utterance_id`, and a `run_id` set once at process boot -- so events from
different provider configs across restarts land as distinguishable runs in
one dataset, and the dashboard's cross-provider comparison view has
something to read even though sessions themselves are in-memory-only.

Perf note (from /plan-eng-review, D7): this is a single-worker asyncio
server where every other codepath deliberately never blocks the event loop.
Appends are small, sequential writes and stay synchronous; the *read* path
is the one that must not block -- `read_recent_runs()` wraps its file parse
in `asyncio.to_thread()` and is tail-bounded (last N lines/runs), since the
file grows unbounded over a multi-week build and an unbounded synchronous
parse on every dashboard request would be exactly the hot-path block this
design otherwise avoids.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_METRICS_PATH = Path(__file__).resolve().parent.parent / "data" / "metrics-runs.jsonl"
DEFAULT_TAIL_LINES = 500


def new_run_id() -> str:
    """One call per process boot -- see module docstring. Not called per
    event; `MetricsLog` calls this once at construction."""
    return uuid.uuid4().hex


@dataclass(frozen=True, slots=True)
class StageEvent:
    """Mirrors the `metrics.stage` WS payload (`stage`, `t_server_ms`,
    `duration_ms`, `utterance_id`, `provider`), plus the two fields needed
    only for cross-restart persistence (`role`, `run_id`) -- see design doc
    `### Metrics / TTFA definition` and `### Metrics persistence`."""

    stage: str
    t_server_ms: int
    duration_ms: float
    utterance_id: str
    provider: str
    role: str
    run_id: str

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


class MetricsLog:
    """One instance per process. `run_id` is generated once at construction
    (process boot), not per event, so every event from one process run
    shares a `run_id` even across a multi-restart file history."""

    def __init__(self, path: Path = DEFAULT_METRICS_PATH, run_id: str | None = None) -> None:
        self.path = Path(path)
        self.run_id = run_id or new_run_id()

    def record(
        self,
        *,
        stage: str,
        duration_ms: float,
        utterance_id: str,
        provider: str,
        role: str,
        t_server_ms: int | None = None,
    ) -> StageEvent:
        """Append one stage event as a JSON line.

        Synchronous: appends are small, sequential writes, not the codepath
        the perf note above is concerned with -- only `read_recent_runs`
        (the dashboard's cross-provider comparison read) is.
        """
        event = StageEvent(
            stage=stage,
            t_server_ms=t_server_ms if t_server_ms is not None else _now_ms(),
            duration_ms=duration_ms,
            utterance_id=utterance_id,
            provider=provider,
            role=role,
            run_id=self.run_id,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(event.to_json_line())
            f.write("\n")
        return event

    async def read_recent_runs(self, max_lines: int = DEFAULT_TAIL_LINES) -> list[dict]:
        """Tail-bounded read of the last `max_lines` events, off the event
        loop thread (see module docstring perf note)."""
        return await asyncio.to_thread(self._read_recent_runs_sync, max_lines)

    def _read_recent_runs_sync(self, max_lines: int) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as f:
            tail = deque(f, maxlen=max_lines)
        events: list[dict] = []
        for line in tail:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
        return events


def _now_ms() -> int:
    return int(time.time() * 1000)
