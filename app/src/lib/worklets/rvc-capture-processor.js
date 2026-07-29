/**
 * RVC realtime capture worklet.
 *
 * Downsamples the microphone (running at the capture AudioContext's sample
 * rate) to 16 kHz mono f32 with a streaming linear resampler and posts
 * fixed-size blocks (block_frame_16k samples) to the main thread, which relays
 * them over the WebSocket as raw little-endian f32 PCM.
 *
 * Allocation discipline: the accumulator is preallocated once in the
 * constructor. Nothing is allocated in the per-render-quantum hot path. A single
 * block-sized Float32Array is allocated only when a full block (~block_ms, e.g.
 * every 300 ms) is ready, then transferred — unavoidable for a postMessage
 * hand-off and orders of magnitude rarer than the 128-sample render quantum.
 *
 * Resampler: a push-based linear interpolator. `step = ctxRate / 16000` input
 * samples advance per emitted output sample; `frac` carries the sub-sample
 * phase across render quanta so there is no discontinuity at quantum joins.
 * When ctxRate === 16000 (the common case, since the main thread requests a
 * 16 kHz capture context) this degenerates to a 1:1 pass-through.
 */
class RvcCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = options?.processorOptions || {};
    this._blockSize = opts.blockSize | 0;
    const targetRate = opts.targetRate || 16000;
    // Input samples consumed per emitted 16 kHz sample.
    this._step = sampleRate / targetRate;
    this._acc = new Float32Array(this._blockSize);
    this._accLen = 0;
    this._frac = 0;
    this._last = 0;
    this._running = this._blockSize > 0;
    this.port.onmessage = (event) => {
      if (event.data && event.data.type === 'stop') {
        this._running = false;
      }
    };
  }

  process(inputs) {
    if (!this._running) {
      return true;
    }
    const input = inputs[0];
    if (!input || input.length === 0) {
      return true;
    }
    const channel = input[0];
    if (!channel) {
      return true;
    }

    const step = this._step;
    const acc = this._acc;
    const size = this._blockSize;
    let accLen = this._accLen;
    let frac = this._frac;
    let last = this._last;

    for (let i = 0; i < channel.length; i++) {
      const x = channel[i];
      while (frac < 1) {
        acc[accLen++] = last + (x - last) * frac;
        frac += step;
        if (accLen === size) {
          const out = new Float32Array(size);
          out.set(acc);
          this.port.postMessage(out, [out.buffer]);
          accLen = 0;
        }
      }
      frac -= 1;
      last = x;
    }

    this._accLen = accLen;
    this._frac = frac;
    this._last = last;
    return true;
  }
}

registerProcessor('rvc-capture-processor', RvcCaptureProcessor);
