/**
 * Connection store -- WS connection lifecycle state, reconnect backoff, and
 * playback_generation. Mirrors design doc `### Reconnect & heartbeat` and
 * `### playback_generation ownership`.
 */

import { create } from "zustand";

export type ConnectionStatus = "disconnected" | "connecting" | "connected" | "recovering";

// Reconnect backoff: 0.5s -> 1 -> 2 -> 4 -> 8s cap, unlimited attempts while
// the tab is open (design doc Reconnect & heartbeat table).
const RECONNECT_BACKOFF_MS = [500, 1000, 2000, 4000, 8000];

// SESSION_NOT_FOUND or N failed session.resume attempts -> stop retrying,
// fall back to a fresh session.start (design doc D8 / T2).
export const MAX_RESUME_ATTEMPTS = 5;

export function reconnectDelayMs(attempt: number): number {
  const index = Math.min(attempt, RECONNECT_BACKOFF_MS.length - 1);
  return RECONNECT_BACKOFF_MS[index];
}

interface ConnectionState {
  status: ConnectionStatus;
  /** Client-owned, monotonic (u16 wraps in the wire encoding). Bumped on
   * (1) local barge-in/Esc mute, (2) start of each new assistant Speaking
   * turn. Server never invents this -- it only echoes the last client
   * value. Client drops any inbound audio with generation < this value. */
  playbackGeneration: number;
  resumeAttempts: number;

  setStatus: (status: ConnectionStatus) => void;
  bumpPlaybackGeneration: () => number;
  incrementResumeAttempts: () => number;
  resetResumeAttempts: () => void;
}

export const useConnectionStore = create<ConnectionState>((set, get) => ({
  status: "disconnected",
  playbackGeneration: 0,
  resumeAttempts: 0,

  setStatus: (status) => set({ status }),

  bumpPlaybackGeneration: () => {
    // u16 wrap, per the wire format (offset 13: u16 playback_generation).
    const next = (get().playbackGeneration + 1) & 0xffff;
    set({ playbackGeneration: next });
    return next;
  },

  incrementResumeAttempts: () => {
    const next = get().resumeAttempts + 1;
    set({ resumeAttempts: next });
    return next;
  },

  resetResumeAttempts: () => set({ resumeAttempts: 0 }),
}));
