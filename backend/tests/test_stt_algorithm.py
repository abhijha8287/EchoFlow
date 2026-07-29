"""STT capability-based branching tests — see `backend/stt/algorithm.py` and
design doc `### Turn-taking policy` (STT algorithm) / D3 (T4).

No pytest-asyncio dependency is added for this: each async test body is
driven via `asyncio.run()` from a plain sync test function.
"""

from __future__ import annotations

import asyncio

import pytest

from stt.algorithm import (
    MAX_CHUNK_INTERVAL_S,
    MIN_CHUNK_INTERVAL_S,
    run_chunked_stt,
    run_stt,
    run_streaming_stt,
)
from stt.provider import RateLimitedError


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
        self.received_pcm: list[bytes] = []

    async def transcribe_chunk(self, utterance_id: str, chunk_index: int, pcm: bytes) -> str:
        if self._in_flight:
            raise AssertionError("transcribe_chunk called while a previous call was still in flight")
        self._in_flight = True
        self.start_times.append(asyncio.get_running_loop().time())
        self.received_pcm.append(pcm)
        try:
            if self._response_delay_s:
                await asyncio.sleep(self._response_delay_s)
            text = self._transcripts[min(chunk_index, len(self._transcripts) - 1)]
            self.call_count += 1
            return text
        finally:
            self._in_flight = False


class FakeRateLimitedBatchProvider(FakeBatchProvider):
    """Batch provider that returns a 429-equivalent on the Nth call."""

    def __init__(self, transcripts: list[str], fail_on_call_index: int) -> None:
        super().__init__(transcripts, response_delay_s=0.0)
        self._fail_on_call_index = fail_on_call_index

    async def transcribe_chunk(self, utterance_id: str, chunk_index: int, pcm: bytes) -> str:
        if chunk_index == self._fail_on_call_index:
            raise RateLimitedError("429")
        return await super().transcribe_chunk(utterance_id, chunk_index, pcm)


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

    result = await run_chunked_stt(provider, "utt-1", on_partial, endpoint, lambda: b"fake-pcm", min_interval_s=0.01)

    # FakeBatchProvider itself raises if re-entered while in flight, so
    # reaching here at all already proves at most 1 request in flight; this
    # is a second, independent check on the recorded call count. 4, not 3:
    # the 3 in-loop calls plus the unconditional finalize call made once
    # `endpoint` is set (see run_chunked_stt docstring) -- the finalize call
    # doesn't invoke on_partial again, so `partials` stays at 3.
    assert provider.call_count == 4
    assert partials == transcripts
    assert result == transcripts[-1]
    assert provider.received_pcm == [b"fake-pcm"] * 4


def test_chunked_stt_does_not_start_next_chunk_before_min_interval_elapsed():
    asyncio.run(_chunked_stt_does_not_start_next_chunk_before_min_interval_elapsed())


async def _chunked_stt_does_not_start_next_chunk_before_min_interval_elapsed():
    # Response resolves near-instantly; without the interval floor the old
    # (buggy) behavior would fire the next chunk almost immediately, i.e. at
    # a much shorter cadence than intended.
    provider = FakeBatchProvider(["a", "b", "c", "d"], response_delay_s=0.0)
    endpoint = asyncio.Event()
    interval = 0.05  # scaled down from the real 0.8-1.0s default for test speed

    await run_chunked_stt(provider, "utt-1", _stop_after_n(endpoint, 3), endpoint, lambda: b"", min_interval_s=interval)

    # 4 calls: the 3 in-loop ones plus the unconditional finalize call made
    # right as `endpoint` becomes set (see run_chunked_stt docstring). Only
    # the first 3 (in-loop) starts are interval-gated -- the finalize call
    # deliberately fires immediately, since nothing follows it for this
    # utterance and there's no pileup risk left to guard against.
    assert len(provider.start_times) == 4
    in_loop_starts = provider.start_times[:3]
    gaps = [b - a for a, b in zip(in_loop_starts, in_loop_starts[1:])]
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
        provider, "utt-1", _stop_after_n(endpoint, len(transcripts)), endpoint, lambda: b"", min_interval_s=0.01
    )

    assert result == transcripts[-1]


def test_chunked_stt_transcribes_even_a_very_short_utterance_with_zero_in_loop_calls():
    asyncio.run(_chunked_stt_transcribes_even_a_very_short_utterance_with_zero_in_loop_calls())


async def _chunked_stt_transcribes_even_a_very_short_utterance_with_zero_in_loop_calls():
    """A very short utterance can reach the VAD endpoint before the
    in-loop `while not endpoint.is_set()` check ever runs (0 iterations).
    Without the unconditional finalize call, that audio would never reach
    the provider at all -- a silently empty transcript for real speech."""
    provider = FakeBatchProvider(["hi there"], response_delay_s=0.0)
    endpoint = asyncio.Event()
    endpoint.set()  # already at endpoint before run_chunked_stt is even called

    result = await run_chunked_stt(provider, "utt-1", lambda _: None, endpoint, lambda: b"short-utterance-pcm")

    assert provider.call_count == 1
    assert provider.received_pcm == [b"short-utterance-pcm"]
    assert result == "hi there"


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

    streaming_result = await run_stt(streaming_provider, "utt-1", lambda _: None, asyncio.Event(), lambda: b"")
    assert streaming_result == "one two"

    batch_endpoint = asyncio.Event()
    batch_result = await run_stt(batch_provider, "utt-2", _stop_after_n(batch_endpoint, 1), batch_endpoint, lambda: b"")
    assert batch_result == "x"


def test_rate_limited_error_surfaces_cleanly_with_no_request_pileup():
    asyncio.run(_rate_limited_error_surfaces_cleanly_with_no_request_pileup())


async def _rate_limited_error_surfaces_cleanly_with_no_request_pileup():
    """T7: a 429 on the batch endpoint must propagate as RateLimitedError
    -- no swallow-and-retry loop that could pile requests up behind the
    failure, which the capped-concurrency design (T4/D3) intentionally does
    not provide."""
    provider = FakeRateLimitedBatchProvider(["a", "b"], fail_on_call_index=1)
    endpoint = asyncio.Event()

    with pytest.raises(RateLimitedError):
        await run_chunked_stt(provider, "utt-1", lambda _: None, endpoint, lambda: b"", min_interval_s=0.01)

    # Exactly the calls up to and including the failing one -- nothing
    # queued or retried behind it.
    assert provider.call_count == 1  # chunk 0 succeeded and incremented call_count
    assert len(provider.start_times) == 1  # chunk 1 raised before recording a start -- nothing else in flight
