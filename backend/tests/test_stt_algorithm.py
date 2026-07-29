"""STT capability-based branching tests — see `backend/stt/algorithm.py` and
design doc `### Turn-taking policy` (STT algorithm) / D3 (T4).

No pytest-asyncio dependency is added for this: each async test body is
driven via `asyncio.run()` from a plain sync test function.
"""

from __future__ import annotations

import asyncio

from stt.algorithm import (
    MAX_CHUNK_INTERVAL_S,
    MIN_CHUNK_INTERVAL_S,
    run_chunked_stt,
    run_stt,
    run_streaming_stt,
)


class FakeBatchProvider:
    """Batch/chunked-only fake -- supports_partial_streaming == False.

    Tracks whether a call is currently in flight (raising if re-entered
    concurrently) and the wall-clock time each chunk request started, so
    tests can assert both the capped-concurrency and interval guarantees
    from the design doc.
    """

    supports_partial_streaming = False

    def __init__(self, transcripts: list[str], response_delay_s: float = 0.0) -> None:
        self._transcripts = transcripts
        self._response_delay_s = response_delay_s
        self._in_flight = False
        self.start_times: list[float] = []
        self.call_count = 0

    async def transcribe_chunk(self, utterance_id: str, chunk_index: int) -> str:
        if self._in_flight:
            raise AssertionError("transcribe_chunk called while a previous call was still in flight")
        self._in_flight = True
        self.start_times.append(asyncio.get_running_loop().time())
        try:
            if self._response_delay_s:
                await asyncio.sleep(self._response_delay_s)
            text = self._transcripts[min(chunk_index, len(self._transcripts) - 1)]
            self.call_count += 1
            return text
        finally:
            self._in_flight = False


class FakeStreamingProvider:
    """True-streaming fake -- supports_partial_streaming == True."""

    supports_partial_streaming = True

    def __init__(self, partials: list[str]) -> None:
        self._partials = partials

    async def stream_partials(self, utterance_id: str):
        for partial in self._partials:
            yield partial


def _stop_after_n(endpoint: asyncio.Event, n: int):
    """Returns an `on_partial`-shaped callback that sets `endpoint` the Nth
    time it's called. Deterministic (no racing background task): the
    algorithm checks `endpoint.is_set()` synchronously right after each
    `on_partial` call, so the loop always stops at exactly N chunks."""
    count = 0

    def on_partial(_: str) -> None:
        nonlocal count
        count += 1
        if count >= n:
            endpoint.set()

    return on_partial


def test_chunk_interval_constants_are_within_800ms_to_1s():
    """D3: the fix widens the interval from a fixed 300ms clock to the
    800ms-1s range sized to a batch endpoint's typical response time."""
    assert MIN_CHUNK_INTERVAL_S >= 0.8
    assert MAX_CHUNK_INTERVAL_S <= 1.0
    assert MIN_CHUNK_INTERVAL_S <= MAX_CHUNK_INTERVAL_S


def test_chunked_stt_never_has_more_than_one_request_in_flight():
    asyncio.run(_chunked_stt_never_has_more_than_one_request_in_flight())


async def _chunked_stt_never_has_more_than_one_request_in_flight():
    transcripts = ["hello", "hello world", "hello world done"]
    provider = FakeBatchProvider(transcripts, response_delay_s=0.02)
    endpoint = asyncio.Event()
    partials: list[str] = []

    def on_partial(text: str) -> None:
        partials.append(text)
        if len(partials) >= 3:
            endpoint.set()

    result = await run_chunked_stt(provider, "utt-1", on_partial, endpoint, min_interval_s=0.01)

    # FakeBatchProvider itself raises if re-entered while in flight, so
    # reaching here at all already proves at most 1 request in flight; this
    # is a second, independent check on the recorded call count.
    assert provider.call_count == 3
    assert partials == transcripts
    assert result == transcripts[-1]


def test_chunked_stt_does_not_start_next_chunk_before_min_interval_elapsed():
    asyncio.run(_chunked_stt_does_not_start_next_chunk_before_min_interval_elapsed())


async def _chunked_stt_does_not_start_next_chunk_before_min_interval_elapsed():
    # Response resolves near-instantly; without the interval floor the old
    # (buggy) behavior would fire the next chunk almost immediately, i.e. at
    # a much shorter cadence than intended.
    provider = FakeBatchProvider(["a", "b", "c", "d"], response_delay_s=0.0)
    endpoint = asyncio.Event()
    interval = 0.05  # scaled down from the real 0.8-1.0s default for test speed

    await run_chunked_stt(provider, "utt-1", _stop_after_n(endpoint, 3), endpoint, min_interval_s=interval)

    assert len(provider.start_times) == 3
    gaps = [b - a for a, b in zip(provider.start_times, provider.start_times[1:])]
    for gap in gaps:
        assert gap >= interval - 0.01  # small tolerance for measurement noise; asyncio timers don't fire early


def test_chunked_stt_returns_last_partial_unambiguously():
    asyncio.run(_chunked_stt_returns_last_partial_unambiguously())


async def _chunked_stt_returns_last_partial_unambiguously():
    """Because concurrency is capped to 1, "last received" and "last sent"
    are always the same response -- there is nothing to disambiguate."""
    transcripts = ["order", "order a", "order a large"]
    provider = FakeBatchProvider(transcripts, response_delay_s=0.01)
    endpoint = asyncio.Event()

    result = await run_chunked_stt(
        provider, "utt-1", _stop_after_n(endpoint, len(transcripts)), endpoint, min_interval_s=0.01
    )

    assert result == transcripts[-1]


def test_streaming_stt_forwards_partials_and_returns_last():
    asyncio.run(_streaming_stt_forwards_partials_and_returns_last())


async def _streaming_stt_forwards_partials_and_returns_last():
    provider = FakeStreamingProvider(["h", "he", "hel", "hello"])
    partials: list[str] = []

    result = await run_streaming_stt(provider, "utt-1", partials.append)

    assert partials == ["h", "he", "hel", "hello"]
    assert result == "hello"


def test_run_stt_dispatches_on_capability_flag_not_vendor_identity():
    asyncio.run(_run_stt_dispatches_on_capability_flag_not_vendor_identity())


async def _run_stt_dispatches_on_capability_flag_not_vendor_identity():
    streaming_provider = FakeStreamingProvider(["one", "one two"])
    batch_provider = FakeBatchProvider(["x"], response_delay_s=0.0)

    streaming_result = await run_stt(streaming_provider, "utt-1", lambda _: None, asyncio.Event())
    assert streaming_result == "one two"

    batch_endpoint = asyncio.Event()
    batch_result = await run_stt(batch_provider, "utt-2", _stop_after_n(batch_endpoint, 1), batch_endpoint)
    assert batch_result == "x"
