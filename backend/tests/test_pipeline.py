"""Pipeline orchestrator tests — see `backend/voice/pipeline.py` and design
doc `### Cancel scope boundary` (T3).

No pytest-asyncio dependency is added for this: each async test body is
driven via `asyncio.run()` from a plain sync test function.
"""

from __future__ import annotations

import asyncio

import pytest

from voice.pipeline import run_utterance_pipeline


def test_stages_run_in_sequence_with_expected_arguments():
    asyncio.run(_stages_run_in_sequence_with_expected_arguments())


async def _stages_run_in_sequence_with_expected_arguments():
    calls: list[str] = []

    async def stt(utterance_id: str) -> str:
        calls.append(f"stt:{utterance_id}")
        return "hello world"

    async def llm(utterance_id: str, transcript: str) -> str:
        calls.append(f"llm:{utterance_id}:{transcript}")
        return "hi there"

    async def tts(utterance_id: str, response: str) -> None:
        calls.append(f"tts:{utterance_id}:{response}")

    await run_utterance_pipeline("utt-1", stt, llm, tts)

    assert calls == [
        "stt:utt-1",
        "llm:utt-1:hello world",
        "tts:utt-1:hi there",
    ]


def test_cancellation_stops_the_pipeline_without_running_later_stages():
    asyncio.run(_cancellation_stops_the_pipeline_without_running_later_stages())


async def _cancellation_stops_the_pipeline_without_running_later_stages():
    """Guards the cancel scope boundary contract from the pipeline's own
    side: cancelling the owning task must unwind whichever stage is in
    flight and must never let a later stage start."""
    calls: list[str] = []
    stt_started = asyncio.Event()

    async def stt(utterance_id: str) -> str:
        calls.append("stt")
        stt_started.set()
        await asyncio.Event().wait()  # hang until cancelled, like a real in-flight call
        return "unreachable"

    async def llm(utterance_id: str, transcript: str) -> str:
        calls.append("llm")
        return "unreachable"

    async def tts(utterance_id: str, response: str) -> None:
        calls.append("tts")

    task = asyncio.create_task(run_utterance_pipeline("utt-1", stt, llm, tts))
    await stt_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert calls == ["stt"]
