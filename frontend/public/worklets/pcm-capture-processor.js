// AudioWorklet processor: captures mono mic audio and posts s16le PCM
// frames to the main thread in ~20ms chunks (320 samples @ 16kHz), per the
// design doc's WebSocket wire format ("Max binary PCM payload ~20-40ms of
// audio per frame").
//
// Runs in AudioWorkletGlobalScope: no imports, no TypeScript, no bundler --
// loaded directly via audioContext.audioWorklet.addModule(). Must stay
// small and non-blocking; the render thread has a hard real-time budget.
//
// Deliberately does NOT resample. `sampleRate` here is whatever the
// AudioContext that registered this worklet was constructed with -- the
// caller (hooks/useMicrophone.ts) is responsible for requesting a 16000Hz
// context and refusing to start capture if the browser didn't honor it, so
// this processor can assume its input already IS 16kHz rather than trying
// to resample on the real-time thread.

const FRAME_SAMPLES = 320; // 20ms @ 16kHz

class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = new Int16Array(FRAME_SAMPLES);
    this._writeIndex = 0;
  }

  process(inputs) {
    const input = inputs[0];
    const channel = input && input[0];
    if (!channel) {
      return true;
    }

    for (let i = 0; i < channel.length; i++) {
      const clamped = Math.max(-1, Math.min(1, channel[i]));
      this._buffer[this._writeIndex++] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;

      if (this._writeIndex === FRAME_SAMPLES) {
        const frame = this._buffer.buffer.slice(0);
        this.port.postMessage(frame, [frame]);
        this._writeIndex = 0;
      }
    }

    return true;
  }
}

registerProcessor("pcm-capture-processor", PcmCaptureProcessor);
