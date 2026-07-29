/**
 * RVC realtime playback worklet.
 *
 * Consumes converted PCM (at the model sample rate) posted from the main thread
 * into a preallocated ring buffer, applies a small jitter buffer (`primeSamples`,
 * ~1 block), and writes to the output. If the playback AudioContext runs at a
 * different rate than the model (it normally does not — the main thread requests
 * a context at model_sr), a streaming linear resampler converts model rate ->
 * context rate on the fly.
 *
 * Allocation discipline: the ring buffer AND the outbound stats message are
 * preallocated once. process() and the inbound message handler only touch
 * scalars and the ring — nothing is allocated per render quantum or per inbound
 * block. The stats message object is reused (postMessage structured-clones it at
 * send time, so mutating it afterwards is safe).
 *
 * Backpressure & latency control: the ring is bounded. If the producer outruns
 * the consumer the oldest samples are dropped rather than growing memory. On
 * underrun the output is silenced and the jitter buffer re-primes before
 * resuming, trading a brief gap for click-free resumption instead of stuttering
 * on a starved buffer.
 *
 * Drain policy: re-priming on underrun refills the buffer, but a transient
 * overload can leave the ring PERMANENTLY over-filled (up to the whole ring),
 * which is permanent added latency. So when the buffer stays above
 * `target + one block` continuously for `drainHoldFrames` output frames, the
 * oldest samples are dropped in one step back down to `target` (the prime
 * level). Dropping (rather than growing) trades one rare, brief seam for bounded
 * mouth-to-ear latency. The current buffered depth (ms at the model rate) is
 * reported to the main thread on a throttled cadence so the UI can show it.
 */
class RvcPlaybackProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = options?.processorOptions || {};
    this._size = Math.max(1, opts.ringSize | 0);
    this._ring = new Float32Array(this._size);
    this._read = 0;
    this._write = 0;
    this._count = 0;
    this._prime = Math.max(1, opts.primeSamples | 0);
    this._primed = false;
    const inputRate = opts.inputRate || sampleRate;
    // Input samples consumed per emitted output sample.
    this._ratio = inputRate / sampleRate;
    this._inputRate = inputRate;
    this._pos = 0;
    this._s0 = 0;
    this._s1 = 0;

    // Drain policy state. `target` is the depth we re-prime to and drain back
    // to; `drainHigh` is one block above it. `overFrames` counts how long
    // (in output frames) the buffer has been continuously over `drainHigh`.
    this._target = this._prime;
    const block = Math.max(1, opts.blockSamples | 0) || this._prime;
    this._drainHigh = this._target + block;
    const drainHoldMs = opts.drainHoldMs > 0 ? opts.drainHoldMs : 250;
    this._drainHoldFrames = Math.max(1, ((drainHoldMs / 1000) * sampleRate) | 0);
    this._overFrames = 0;

    // Throttled buffered-depth reporting (preallocated message, reused).
    this._reportFrames = Math.max(1, ((sampleRate * 0.1) | 0));
    this._sinceReport = 0;
    this._stat = { type: 'buffered', ms: 0 };

    this.port.onmessage = (event) => {
      const pcm = event.data?.pcm ? event.data.pcm : null;
      if (!pcm) {
        return;
      }
      const ring = this._ring;
      const size = this._size;
      let write = this._write;
      let count = this._count;
      let read = this._read;
      for (let i = 0; i < pcm.length; i++) {
        ring[write] = pcm[i];
        write = (write + 1) % size;
        if (count < size) {
          count++;
        } else {
          read = (read + 1) % size;
        }
      }
      this._write = write;
      this._count = count;
      this._read = read;
    };
  }

  _reportBuffered(frames) {
    this._sinceReport += frames;
    if (this._sinceReport < this._reportFrames) {
      return;
    }
    this._sinceReport = 0;
    const stat = this._stat;
    stat.ms = (this._count / this._inputRate) * 1000;
    this.port.postMessage(stat);
  }

  process(_inputs, outputs) {
    const output = outputs[0];
    if (!output || output.length === 0) {
      return true;
    }
    const out = output[0];
    const frames = out.length;

    this._reportBuffered(frames);

    if (!this._primed) {
      if (this._count < this._prime) {
        out.fill(0);
        for (let c = 1; c < output.length; c++) {
          output[c].set(out);
        }
        return true;
      }
      this._primed = true;
    }

    const ring = this._ring;
    const size = this._size;
    let read = this._read;
    let count = this._count;

    // Drain policy: if the ring has been over-filled for long enough, drop the
    // oldest samples down to the prime target so a past overload does not leave
    // permanent latency. Runs at most once per callback and only after the
    // buffer has stayed above the high-water mark continuously.
    if (count > this._drainHigh) {
      this._overFrames += frames;
      if (this._overFrames >= this._drainHoldFrames) {
        const drop = count - this._target;
        read = (read + drop) % size;
        count -= drop;
        this._overFrames = 0;
      }
    } else {
      this._overFrames = 0;
    }

    let pos = this._pos;
    let s0 = this._s0;
    let s1 = this._s1;

    for (let k = 0; k < frames; k++) {
      while (pos >= 1) {
        if (count <= 0) {
          // Underrun: silence the remainder and re-prime before resuming.
          for (let j = k; j < frames; j++) {
            out[j] = 0;
          }
          this._read = read;
          this._count = count;
          this._pos = pos;
          this._s0 = s0;
          this._s1 = s1;
          this._primed = false;
          this._overFrames = 0;
          for (let c = 1; c < output.length; c++) {
            output[c].set(out);
          }
          return true;
        }
        s0 = s1;
        s1 = ring[read];
        read = (read + 1) % size;
        count--;
        pos -= 1;
      }
      out[k] = s0 + (s1 - s0) * pos;
      pos += this._ratio;
    }

    this._read = read;
    this._count = count;
    this._pos = pos;
    this._s0 = s0;
    this._s1 = s1;
    for (let c = 1; c < output.length; c++) {
      output[c].set(out);
    }
    return true;
  }
}

registerProcessor('rvc-playback-processor', RvcPlaybackProcessor);
