"""Metrics persistence tests — see `backend/services/metrics.py` and design
doc `### Metrics persistence` (T5).

No pytest-asyncio dependency is added for this: each async test body is
driven via `asyncio.run()` from a plain sync test function.
"""

from __future__ import annotations

import asyncio
import threading

from services.metrics import MetricsLog


def test_prior_run_entries_persist_across_a_simulated_restart(tmp_path):
    asyncio.run(_prior_run_entries_persist_across_a_simulated_restart(tmp_path))


async def _prior_run_entries_persist_across_a_simulated_restart(tmp_path):
    """Two `MetricsLog` instances pointed at the same file with different
    run_ids simulate two process boots. Both runs' events must still be
    readable, correctly tagged with `provider`/`run_id`, after the
    "restart" -- this is the persistence gap the design (D2) closes."""
    path = tmp_path / "metrics-runs.jsonl"

    first_boot = MetricsLog(path=path, run_id="run-gemini")
    first_boot.record(
        stage="stt.duration_ms", duration_ms=120.0, utterance_id="u1", provider="gemini", role="stt"
    )
    first_boot.record(
        stage="llm.first_token_ms", duration_ms=300.0, utterance_id="u1", provider="gemini", role="llm"
    )

    # Simulate a restart with a different provider config: fresh instance,
    # fresh run_id, same file -- in-memory session state would be wiped,
    # but the file must not be.
    second_boot = MetricsLog(path=path, run_id="run-openai")
    second_boot.record(
        stage="stt.duration_ms", duration_ms=450.0, utterance_id="u2", provider="openai", role="stt"
    )

    events = await second_boot.read_recent_runs()

    assert len(events) == 3
    run_ids = {e["run_id"] for e in events}
    assert run_ids == {"run-gemini", "run-openai"}

    gemini_events = [e for e in events if e["run_id"] == "run-gemini"]
    assert {e["provider"] for e in gemini_events} == {"gemini"}
    assert {e["stage"] for e in gemini_events} == {"stt.duration_ms", "llm.first_token_ms"}

    openai_events = [e for e in events if e["run_id"] == "run-openai"]
    assert len(openai_events) == 1
    assert openai_events[0]["provider"] == "openai"
    assert openai_events[0]["duration_ms"] == 450.0
    assert openai_events[0]["utterance_id"] == "u2"
    assert openai_events[0]["role"] == "stt"


def test_read_recent_runs_is_tail_bounded(tmp_path):
    asyncio.run(_read_recent_runs_is_tail_bounded(tmp_path))


async def _read_recent_runs_is_tail_bounded(tmp_path):
    path = tmp_path / "metrics-runs.jsonl"
    log = MetricsLog(path=path, run_id="run-a")
    for i in range(10):
        log.record(
            stage="stt.duration_ms", duration_ms=float(i), utterance_id=f"u{i}", provider="gemini", role="stt"
        )

    events = await log.read_recent_runs(max_lines=3)

    assert len(events) == 3
    # Must be the *last* 3 written, in original order, not an arbitrary 3.
    assert [e["utterance_id"] for e in events] == ["u7", "u8", "u9"]


def test_read_recent_runs_on_missing_file_returns_empty(tmp_path):
    asyncio.run(_read_recent_runs_on_missing_file_returns_empty(tmp_path))


async def _read_recent_runs_on_missing_file_returns_empty(tmp_path):
    log = MetricsLog(path=tmp_path / "does-not-exist.jsonl", run_id="run-a")
    assert await log.read_recent_runs() == []


def test_read_recent_runs_parses_off_the_event_loop_thread(tmp_path):
    asyncio.run(_read_recent_runs_parses_off_the_event_loop_thread(tmp_path))


async def _read_recent_runs_parses_off_the_event_loop_thread(tmp_path):
    """Perf note (D7): the read path must not block the single-worker event
    loop. Confirmed behaviorally, not just structurally -- the sync parse
    must actually execute on a different thread than the event loop."""
    path = tmp_path / "metrics-runs.jsonl"
    log = MetricsLog(path=path, run_id="run-a")
    log.record(stage="stt.duration_ms", duration_ms=1.0, utterance_id="u1", provider="gemini", role="stt")

    event_loop_thread_id = threading.get_ident()
    seen_thread_ids: list[int] = []
    original_read_sync = log._read_recent_runs_sync

    def spying_read_sync(max_lines: int):
        seen_thread_ids.append(threading.get_ident())
        return original_read_sync(max_lines)

    log._read_recent_runs_sync = spying_read_sync
    events = await log.read_recent_runs()

    assert len(events) == 1
    assert len(seen_thread_ids) == 1
    assert seen_thread_ids[0] != event_loop_thread_id
