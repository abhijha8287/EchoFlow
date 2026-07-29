"use client";

/**
 * P0a-i "basic state text UI" -- proves the loop end to end (FSM, WS
 * protocol, VAD endpointing, barge-in) without the P0b orb/waveform/
 * dashboard polish, per the design doc's milestone sequencing.
 */

import { useState } from "react";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";
import { useVoiceSession } from "@/hooks/useVoiceSession";
import { useConnectionStore } from "@/stores/connection";
import type { SessionMode } from "@/stores/session";
import { useSessionStore } from "@/stores/session";

export default function Home() {
  const [mode, setMode] = useState<SessionMode>("continuous");
  const [connected, setConnected] = useState(false);

  const voice = useVoiceSession();
  const fsmState = useSessionStore((s) => s.fsmState);
  const transcriptPartial = useSessionStore((s) => s.transcriptPartial);
  const transcriptFinal = useSessionStore((s) => s.transcriptFinal);
  const llmResponse = useSessionStore((s) => s.llmResponse);
  const lastError = useSessionStore((s) => s.lastError);
  const sessionId = useSessionStore((s) => s.sessionId);
  const connectionStatus = useConnectionStore((s) => s.status);

  useKeyboardShortcuts({
    onInterrupt: voice.interrupt,
    onPttStart: mode === "ptt" ? voice.pttStart : undefined,
    onPttEnd: mode === "ptt" ? voice.pttEnd : undefined,
    enabled: connected,
  });

  const handleToggleConnect = () => {
    if (connected) {
      voice.disconnect();
      setConnected(false);
    } else {
      voice.connect(mode);
      setConnected(true);
    }
  };

  return (
    <main className="flex flex-1 flex-col items-center justify-center gap-6 p-8 font-mono text-sm">
      <h1 className="text-xl font-semibold">EchoFlow</h1>

      <div className="flex items-center gap-3">
        <label className="flex items-center gap-2">
          <span>Mode:</span>
          <select
            value={mode}
            disabled={connected}
            onChange={(e) => setMode(e.target.value as SessionMode)}
            className="rounded border border-current bg-transparent px-2 py-1"
          >
            <option value="continuous">continuous</option>
            <option value="ptt">ptt</option>
          </select>
        </label>

        <button onClick={handleToggleConnect} className="rounded border border-current px-4 py-1 hover:opacity-80">
          {connected ? "Disconnect" : "Connect"}
        </button>

        {connected && (
          <button onClick={voice.interrupt} className="rounded border border-current px-4 py-1 hover:opacity-80">
            Interrupt (Esc)
          </button>
        )}
      </div>

      {mode === "ptt" && connected && <p className="opacity-70">Hold Space to talk</p>}

      <dl className="grid w-full max-w-md grid-cols-[auto_1fr] gap-x-4 gap-y-1">
        <dt className="opacity-60">connection</dt>
        <dd>{connectionStatus}</dd>

        <dt className="opacity-60">state</dt>
        <dd className="font-semibold">{fsmState}</dd>

        <dt className="opacity-60">session_id</dt>
        <dd className="truncate">{sessionId ?? "-"}</dd>

        <dt className="opacity-60">mic</dt>
        <dd>
          {voice.isCapturing ? "capturing" : "idle"}
          {voice.micError && <span className="text-red-500"> -- {voice.micError}</span>}
        </dd>

        <dt className="opacity-60">transcript</dt>
        <dd>{transcriptFinal || transcriptPartial || "-"}</dd>

        <dt className="opacity-60">assistant</dt>
        <dd>{llmResponse || "-"}</dd>

        {lastError && (
          <>
            <dt className="text-red-500">error</dt>
            <dd className="text-red-500">
              [{lastError.code}] {lastError.message}
            </dd>
          </>
        )}
      </dl>
    </main>
  );
}
