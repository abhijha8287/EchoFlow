/**
 * Browser-side PCM <-> Web Audio Float32 conversion. Mirrors the wire
 * format's s16le PCM (see lib/protocol.ts) -- reads the bytes as a
 * platform-native-endian Int16Array, which is little-endian on every
 * real-world browser target (x86/ARM), matching s16le exactly.
 */

export function int16PcmToFloat32(pcm: Uint8Array): Float32Array<ArrayBuffer> {
  const samples = new Int16Array(pcm.buffer, pcm.byteOffset, pcm.byteLength / 2);
  // Explicit ArrayBuffer allocation (not the (buffer, offset, length) view
  // overload) so this is always Float32Array<ArrayBuffer>, matching what
  // AudioBuffer.copyToChannel expects in newer @types/web typings that
  // distinguish it from the more general Float32Array<ArrayBufferLike>.
  const float32 = new Float32Array(new ArrayBuffer(samples.length * Float32Array.BYTES_PER_ELEMENT));
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i];
    float32[i] = s / (s < 0 ? 0x8000 : 0x7fff);
  }
  return float32;
}

// Local barge-in detector (design doc: "Local barge-in detector (energy /
// lightweight VAD)"). Deliberately crude -- a mean-absolute-amplitude
// threshold over one ~20ms frame -- not a real VAD model. Its only job is
// triggering the *optimistic* client-side interrupt while Speaking; the
// server's own VAD remains the source of truth for turn-taking.
const DEFAULT_ENERGY_THRESHOLD = 1500; // out of a possible 0-32767 mean abs amplitude

export function isLikelySpeech(pcm: Int16Array, threshold = DEFAULT_ENERGY_THRESHOLD): boolean {
  if (pcm.length === 0) return false;
  let sum = 0;
  for (let i = 0; i < pcm.length; i++) {
    sum += Math.abs(pcm[i]);
  }
  return sum / pcm.length > threshold;
}
