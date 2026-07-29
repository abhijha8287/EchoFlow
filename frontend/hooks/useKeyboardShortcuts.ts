"use client";

/**
 * Space / Esc, per design doc `### Interaction modes`:
 *   Continuous: Esc interrupts.
 *   PTT: Space held = mic gated open; Esc / button interrupt only.
 */

import { useEffect } from "react";

export interface UseKeyboardShortcutsOptions {
  onInterrupt: () => void;
  onPttStart?: () => void;
  onPttEnd?: () => void;
  enabled?: boolean;
}

export function useKeyboardShortcuts({
  onInterrupt,
  onPttStart,
  onPttEnd,
  enabled = true,
}: UseKeyboardShortcutsOptions): void {
  useEffect(() => {
    if (!enabled) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.repeat) return;
      if (event.code === "Escape") {
        onInterrupt();
      } else if (event.code === "Space" && onPttStart) {
        const target = event.target as HTMLElement | null;
        if (target && ["INPUT", "TEXTAREA"].includes(target.tagName)) return;
        event.preventDefault();
        onPttStart();
      }
    };

    const handleKeyUp = (event: KeyboardEvent) => {
      if (event.code === "Space" && onPttEnd) {
        onPttEnd();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [enabled, onInterrupt, onPttStart, onPttEnd]);
}
