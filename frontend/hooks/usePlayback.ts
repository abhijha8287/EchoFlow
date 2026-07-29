"use client";

/**
 * TTS playback: queues incoming s16le/16k PCM frames and schedules them
 * back-to-back on a single AudioBufferSourceNode chain, per design doc
 * `### playback_generation ownership` and `### Barge-in sequence`.
 *
 * Generation-gated: a frame whose `playbackGeneration` is behind the
 * client's current counter is dropped (server never invents generation, it
 * only echoes the client's last value -- see design doc). `interrupt()`
 * stops every scheduled/in-flight source immediately and tears down the
 * schedule, so barge-in never lets stale audio keep playing (no zombie
 * audio, matching the mid-Speaking reconnect guarantee already covered on
 * the backend in T7).
 */

import { useCallback, useEffect, useRef } from "react";
import { int16PcmToFloat32 } from "@/lib/audio";
import type { BinaryFrame } from "@/lib/protocol";
import { useConnectionStore } from "@/stores/connection";

const PLAYBACK_SAMPLE_RATE = 16000;

export interface UsePlaybackResult {
  enqueue: (frame: BinaryFrame) => void;
  interrupt: () => void;
}

export function usePlayback(): UsePlaybackResult {
  const audioContextRef = useRef<AudioContext | null>(null);
  const nextStartTimeRef = useRef(0);
  const activeSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());

  const getContext = useCallback((): AudioContext => {
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext();
    }
    return audioContextRef.current;
  }, []);

  const enqueue = useCallback(
    (frame: BinaryFrame) => {
      const currentGeneration = useConnectionStore.getState().playbackGeneration;
      if (frame.playbackGeneration < currentGeneration) {
        return; // stale generation -- drop per design doc
      }

      const context = getContext();
      const float32 = int16PcmToFloat32(frame.pcm);
      if (float32.length === 0) return;

      const buffer = context.createBuffer(1, float32.length, PLAYBACK_SAMPLE_RATE);
      buffer.copyToChannel(float32, 0);

      const source = context.createBufferSource();
      source.buffer = buffer;
      source.connect(context.destination);
      source.onended = () => activeSourcesRef.current.delete(source);

      const startAt = Math.max(context.currentTime, nextStartTimeRef.current);
      source.start(startAt);
      nextStartTimeRef.current = startAt + buffer.duration;
      activeSourcesRef.current.add(source);
    },
    [getContext],
  );

  const interrupt = useCallback(() => {
    for (const source of activeSourcesRef.current) {
      try {
        source.stop();
      } catch {
        // Already stopped/ended -- fine, we're tearing down regardless.
      }
    }
    activeSourcesRef.current.clear();
    nextStartTimeRef.current = audioContextRef.current?.currentTime ?? 0;
  }, []);

  useEffect(
    () => () => {
      interrupt();
      void audioContextRef.current?.close();
      audioContextRef.current = null;
    },
    [interrupt],
  );

  return { enqueue, interrupt };
}
