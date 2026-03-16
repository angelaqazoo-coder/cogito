/**
 * audio-processor.js — AudioWorklet that captures raw mic audio,
 * converts Float32 samples to PCM-16 (little-endian), and posts
 * 2048-sample buffers to the main thread for WebSocket transmission.
 *
 * Target: 16000 Hz mono — matches Gemini Live API input requirement.
 */
class PCMProcessor extends AudioWorkletProcessor {
  constructor () {
    super();
    this._buf = [];
    this._chunkSamples = 2048;  // ~128 ms at 16 kHz
  }

  process (inputs) {
    const channel = inputs[0]?.[0];
    if (!channel) return true;

    for (let i = 0; i < channel.length; i++) {
      // Clamp, scale Float32 → Int16
      const s = Math.max(-1, Math.min(1, channel[i]));
      this._buf.push(s < 0 ? s * 0x8000 : s * 0x7fff);
    }

    while (this._buf.length >= this._chunkSamples) {
      const int16 = new Int16Array(this._buf.splice(0, this._chunkSamples));
      // Transfer ownership of the buffer — zero-copy
      this.port.postMessage(int16.buffer, [int16.buffer]);
    }

    return true; // keep processor alive
  }
}

registerProcessor('pcm-processor', PCMProcessor);
