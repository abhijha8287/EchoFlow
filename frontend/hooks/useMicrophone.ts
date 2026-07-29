"use client";

/**
 * Mic capture: getUserMedia + AudioContext(16kHz) + the pcm-capture-
 * processor AudioWorklet (public/worklets/pcm-capture-processor.js).
 *
 * Deliberately fails loud rather than silently sending wrong-rate audio:
 * the server assumes every inbound mic frame is already s16le/16kHz mono
 * (there is no server-side mic-audio resampling, only server-side TTS
 * resampling -- see design doc `### Audio format normalization`), so if
 * the browser won't honor a 16kHz AudioContext, capture must not start.
 */

import { useCallback, useEffect, useRef, useState } from "react";

const TARGET_SAMPLE_RATE = 16000;
const WORKLET_URL = "/worklets/pcm-capture-processor.js";
const WORKLET_NAME = "pcm-capture-processor";

export interface UseMicrophoneResult {
  isCapturing: boolean;
  error: string | null;
  start: () => Promise<void>;
  stop: () => void;
}

export function useMicrophone(onFrame: (pcm: Int16Array) => void): UseMicrophoneResult {
  const [isCapturing, setIsCapturing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const audioContextRef = useRef<AudioContext | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const onFrameRef = useRef(onFrame);
  onFrameRef.current = onFrame;

  const stop = useCallback(() => {
    workletNodeRef.current?.port.close();
    workletNodeRef.current?.disconnect();
    workletNodeRef.current = null;

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    void audioContextRef.current?.close();
    audioContextRef.current = null;

    setIsCapturing(false);
  }, []);

  const start = useCallback(async () => {
    setError(null);
    try {
      const audioContext = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
      if (audioContext.sampleRate !== TARGET_SAMPLE_RATE) {
        const actual = audioContext.sampleRate;
        await audioContext.close();
        throw new Error(
          `Browser did not honor the requested ${TARGET_SAMPLE_RATE}Hz AudioContext sample rate ` +
            `(got ${actual}Hz). The wire protocol requires exactly 16kHz mono PCM -- refusing to ` +
            `capture at the wrong rate rather than silently sending mismatched audio.`,
        );
      }

      await audioContext.audioWorklet.addModule(WORKLET_URL);

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });

      const source = audioContext.createMediaStreamSource(stream);
      const workletNode = new AudioWorkletNode(audioContext, WORKLET_NAME);
      workletNode.port.onmessage = (event: MessageEvent<ArrayBuffer>) => {
        onFrameRef.current(new Int16Array(event.data));
      };
      // Capture only -- deliberately not connected to audioContext.destination
      // (no local mic monitoring/echo).
      source.connect(workletNode);

      audioContextRef.current = audioContext;
      workletNodeRef.current = workletNode;
      streamRef.current = stream;
      setIsCapturing(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setIsCapturing(false);
    }
  }, []);

  useEffect(() => stop, [stop]);

  return { isCapturing, error, start, stop };
}
