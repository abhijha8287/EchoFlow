/**
 * Session store -- Zustand wrapper around the pure FSM in lib/fsm.ts, plus
 * the session-scoped fields the UI needs (session_id, current transcript,
 * mode). Mirrors the backend's Session dataclass (backend/core/session.py)
 * for the fields that matter client-side; connection/reconnect state lives
 * separately in stores/connection.ts.
 */

import { create } from "zustand";
import { InvalidTransitionError, VoiceEvent, VoiceState, transition } from "@/lib/fsm";

export type SessionMode = "continuous" | "ptt";

interface SessionState {
  sessionId: string | null;
  mode: SessionMode;
  fsmState: VoiceState;
  transcriptPartial: string;
  transcriptFinal: string;
  llmResponse: string;
  lastError: { code: string; message: string; retriable: boolean } | null;

  setSessionId: (sessionId: string) => void;
  setMode: (mode: SessionMode) => void;
  dispatch: (event: VoiceEvent) => void;
  /** Force-set the state directly, bypassing transition validation. Used
   * when applying the server's authoritative `state.change` events, which
   * report a state, not the event that produced it -- `dispatch()` is for
   * client-initiated optimistic transitions (e.g. local barge-in), where
   * validating against the table is exactly the point. */
  setFsmState: (state: VoiceState) => void;
  setTranscriptPartial: (text: string) => void;
  setTranscriptFinal: (text: string) => void;
  setLlmResponse: (text: string) => void;
  setError: (error: SessionState["lastError"]) => void;
  reset: () => void;
}

const initialState = {
  sessionId: null as string | null,
  mode: "continuous" as SessionMode,
  fsmState: VoiceState.IDLE,
  transcriptPartial: "",
  transcriptFinal: "",
  llmResponse: "",
  lastError: null as SessionState["lastError"],
};

export const useSessionStore = create<SessionState>((set, get) => ({
  ...initialState,

  setSessionId: (sessionId) => set({ sessionId }),
  setMode: (mode) => set({ mode }),

  dispatch: (event) => {
    const current = get().fsmState;
    try {
      const next = transition(current, event);
      set({ fsmState: next });
    } catch (err) {
      if (err instanceof InvalidTransitionError) {
        // Mirrors the backend's philosophy: an unhandled event is a bug in
        // the caller, not a state to silently mask. Surface it loudly in
        // dev rather than corrupting fsmState.
        console.error(err.message);
        return;
      }
      throw err;
    }
  },

  setFsmState: (state) => set({ fsmState: state }),

  setTranscriptPartial: (text) => set({ transcriptPartial: text }),
  setTranscriptFinal: (text) => set({ transcriptFinal: text, transcriptPartial: "" }),
  setLlmResponse: (text) => set({ llmResponse: text }),
  setError: (error) => set({ lastError: error }),
  reset: () => set(initialState),
}));
