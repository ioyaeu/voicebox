import { useCallback, useEffect, useRef, useState } from 'react';
import type { RvcF0Method } from '@/lib/api/types';
import { useServerStore } from '@/stores/serverStore';

/**
 * Real-time RVC voice-conversion client for the `WS /convert/stream` endpoint.
 *
 * ## WebSocket protocol (must stay in lock-step with `backend/routes/convert.py`)
 *
 * 1. **Handshake — client → server, one TEXT frame (JSON):**
 *    `{ version, profile_id, block_ms, f0_up_key, f0_method, index_rate, rms_mix_rate, protect }`.
 *    `version` is the wire-protocol major (see `STREAM_PROTOCOL_VERSION` in
 *    `backend/models.py`); the server rejects an unrecognised major up front.
 * 2. **Handshake — server → client, one TEXT frame (JSON):**
 *    `{ ready: true, model_sr, block_frame_16k?, block_frame_sr? }` on success, or
 *    `{ error: string }` (and/or a close frame) on failure. When the server omits
 *    the frame sizes they are derived locally from `block_ms` + `model_sr` using
 *    the same 160-sample-hop rounding as the backend session.
 * 3. **Audio — client → server, BINARY frames:** raw little-endian f32 PCM,
 *    16 kHz mono, exactly `block_frame_16k` samples per frame. No header.
 * 4. **Audio — server → client, BINARY frames:** an 8-byte little-endian header
 *    followed by the converted PCM. The header layout is defined once, in the
 *    server module docstring (`backend/routes/convert.py`) as `struct '<fBBH'` —
 *    treat that as the single source of truth and keep the offsets read below in
 *    lock-step with it: `float32 infer_ms` @0, `uint8 overload` @4,
 *    `uint8 dropped` @5, `uint16` reserved @6; the PCM (`block_frame_sr`
 *    `float32` samples at `model_sr`) starts at the 4-byte-aligned offset 8, so
 *    a `Float32Array` view over the payload is valid.
 *
 * The server event loop must never block; on the client the capture and playback
 * resampling both live in AudioWorklets so the main thread only relays bytes and
 * computes cheap per-block meters.
 */

// Mirror backend constants (`_WINDOW`, `_SR_16K`) so client-derived block sizes
// match the session exactly when the server does not send them.
const WORKLET_HOP = 160;
const CAPTURE_RATE = 16000;

// Worklet modules live in `app/src/lib/worklets/` and are referenced with the
// `new URL(..., import.meta.url)` pattern so Vite emits them as hashed assets in
// EVERY consuming build (app/ and web/). The previous pattern joined the app base
// URL with a `public/`-relative path, which only exists in app/ — the web/ build
// has no `public/` dir, so `addModule()` received the SPA's index.html
// ("text/html is not valid JavaScript") and realtime was dead there.
const CAPTURE_WORKLET_URL = new URL('../worklets/rvc-capture-processor.js', import.meta.url).href;
const PLAYBACK_WORKLET_URL = new URL('../worklets/rvc-playback-processor.js', import.meta.url).href;

// Wire-protocol major version the client speaks. Must match
// `STREAM_PROTOCOL_VERSION` in `backend/models.py`; the server rejects an
// unrecognised major in the handshake before loading a model.
const STREAM_PROTOCOL_VERSION = 1;

const SERVER_HEADER_BYTES = 8;
const OUTPUT_JITTER_BLOCKS = 1;
const PLAYBACK_RING_BLOCKS = 8;
// Cap on outstanding capture-send timestamps kept for the latency measurement.
// Bounds memory and stops the mouth-to-ear estimate from ever using an ancient
// timestamp if send/receive correspondence drifts (e.g. after dropped blocks).
const MAX_INFLIGHT_STAMPS = 32;
// Cap of unsent capture data allowed in the socket before we drop the newest
// block client-side (the server drops oldest input on its side). Keeps the
// outbound queue bounded when inference falls behind.
const MAX_SEND_BACKLOG_BLOCKS = 3;
const HANDSHAKE_TIMEOUT_MS = 10000;

export type RvcStreamStatus = 'idle' | 'connecting' | 'running' | 'stopping' | 'error';

export interface RvcStreamConfig {
  profileId: string;
  /** Block size in ms; fixed for the session. Defaults to 300. */
  blockMs?: number;
  f0UpKey: number;
  f0Method: RvcF0Method;
  indexRate: number;
  rmsMixRate: number;
  protect: number;
  /** `deviceId` of the input (microphone); omit for the system default. */
  inputDeviceId?: string;
  /** `deviceId` of the output; omit for the system default. */
  outputDeviceId?: string;
  /** Route converted audio to the local default output too (hear yourself). */
  monitor: boolean;
}

interface Engine {
  ws: WebSocket | null;
  stream: MediaStream | null;
  captureCtx: AudioContext | null;
  captureSource: MediaStreamAudioSourceNode | null;
  captureNode: AudioWorkletNode | null;
  captureSink: GainNode | null;
  playbackCtx: AudioContext | null;
  playbackNode: AudioWorkletNode | null;
  streamDest: MediaStreamAudioDestinationNode | null;
  monitorGain: GainNode | null;
  audioEl: HTMLAudioElement | null;
  blockFrame16k: number;
  blockFrameSr: number;
  blockMs: number;
  modelSr: number;
  maxBacklogBytes: number;
  /**
   * FIFO of `performance.now()` capture-send timestamps, one per binary block
   * sent to the server. The server processes blocks in order, so popping the
   * oldest when a converted block arrives yields a measured send→receive
   * latency. Bounded to `MAX_INFLIGHT_STAMPS`.
   */
  sendTimes: number[];
  stopped: boolean;
}

interface GraphMeters {
  onInputLevel: (level: number) => void;
  onOverload: () => void;
  /**
   * Report the output-routing state. Receives the `deviceId` that could not be
   * reached (so the UI can name it) or `null` when playback is on its intended
   * target (the selected device, or the system default when none was chosen).
   */
  onOutputRouteError: (deviceId: string | null) => void;
  /** Current playback jitter-buffer depth in ms, reported by the worklet. */
  onBufferedMs: (ms: number) => void;
}

// Match Python's built-in `round()` (banker's rounding: half → nearest even) so
// the client-derived fallback block sizes are byte-identical to the backend
// session (`_ms_to_frames` / `round(...)` in `backends/rvc/streaming.py`).
// `Math.round` rounds half up and would disagree on exact-.5 hop counts.
function bankersRound(x: number): number {
  const floor = Math.floor(x);
  const frac = x - floor;
  if (frac < 0.5) return floor;
  if (frac > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

function roundToHop(samples: number): number {
  return Math.max(1, bankersRound(samples / WORKLET_HOP)) * WORKLET_HOP;
}

function computeBlockFrames(blockMs: number, modelSr: number) {
  const blockFrame16k = roundToHop((blockMs / 1000) * CAPTURE_RATE);
  const blockFrameSr = bankersRound((blockFrame16k * modelSr) / CAPTURE_RATE);
  return { blockFrame16k, blockFrameSr };
}

function rms(samples: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) {
    sum += samples[i] * samples[i];
  }
  return Math.sqrt(sum / Math.max(1, samples.length));
}

function toWsUrl(serverUrl: string): string {
  return `${serverUrl.replace(/^http/i, 'ws')}/convert/stream`;
}

function newEngine(): Engine {
  return {
    ws: null,
    stream: null,
    captureCtx: null,
    captureSource: null,
    captureNode: null,
    captureSink: null,
    playbackCtx: null,
    playbackNode: null,
    streamDest: null,
    monitorGain: null,
    audioEl: null,
    blockFrame16k: 0,
    blockFrameSr: 0,
    blockMs: 0,
    modelSr: 0,
    maxBacklogBytes: 0,
    sendTimes: [],
    stopped: false,
  };
}

/**
 * Point the playback `<audio>` element at the requested output device.
 *
 * The `<audio>` element is the route to a SPECIFIC output device. Local playback
 * to the SYSTEM DEFAULT sink is owned entirely by the monitor route
 * (`monitorGain -> playbackCtx.destination`), so that a single toggle controls
 * "hear yourself locally" and there is never a second copy on the same sink.
 * Therefore:
 *
 * - No device requested (system default): MUTE the element. Playing it here
 *   would double up with — and bypass the mute of — the monitor route. This is
 *   not a routing error (the default is always reachable), so it returns `true`.
 * - A NON-DEFAULT device is requested: route to it via `setSinkId`. We must
 *   never silently fall back to the default sink — that would leak the converted
 *   voice to the local speakers while the user believes it is going to (e.g.)
 *   BlackHole for Discord. So if `setSinkId` is missing (unsupported at runtime)
 *   or rejects at play time we mute the element and return `false` so the caller
 *   can warn the user.
 */
async function routeAudioElement(
  audioEl: HTMLAudioElement,
  deviceId: string | undefined,
): Promise<boolean> {
  // No device requested: the monitor route owns default-sink playback, so keep
  // the element muted. Best-effort reset the sink so a later device switch
  // starts from a known state; the default is always a reachable target.
  if (!deviceId) {
    if (typeof audioEl.setSinkId === 'function') {
      try {
        await audioEl.setSinkId('');
      } catch {
        // Resetting to default failed; the monitor route still handles playback.
      }
    }
    audioEl.muted = true;
    return true;
  }
  // A specific device was requested but runtime routing is unavailable: do NOT
  // leak converted audio to the default sink — hold playback (mute) instead.
  if (typeof audioEl.setSinkId !== 'function') {
    audioEl.muted = true;
    return false;
  }
  try {
    await audioEl.setSinkId(deviceId);
    audioEl.muted = false;
    return true;
  } catch {
    // setSinkId rejected at play time (device gone, not permitted, …): hold
    // playback so the user is never misled into thinking the device is fed.
    audioEl.muted = true;
    return false;
  }
}

/**
 * Wire the capture → WS → playback audio graph. Runs once, after the handshake.
 * Capture and playback each get their own AudioContext at the DEFAULT hardware
 * rate. We deliberately do NOT request 16 kHz / model_sr: the Tauri webview
 * (WebKit/WKWebView) is unreliable with non-standard AudioContext rates — it may
 * report the requested rate while actually running the device at 44.1/48 kHz and
 * feeding/emitting unresampled audio, which mis-rates the stream (heard as very
 * slow, deep, deformed playback). Instead the worklets resample against the real
 * `sampleRate` global: capture downsamples hardware -> 16 kHz, playback upsamples
 * model_sr -> hardware (1:1 only when the rates happen to match). So using the
 * default rate is both correct and robust.
 */
async function buildGraph(
  engine: Engine,
  config: RvcStreamConfig,
  meters: GraphMeters,
): Promise<void> {
  const captureCtx = new AudioContext();
  engine.captureCtx = captureCtx;
  await captureCtx.audioWorklet.addModule(CAPTURE_WORKLET_URL);
  if (engine.stopped) return;

  const source = captureCtx.createMediaStreamSource(engine.stream as MediaStream);
  const captureNode = new AudioWorkletNode(captureCtx, 'rvc-capture-processor', {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    channelCount: 1,
    channelCountMode: 'explicit',
    processorOptions: { blockSize: engine.blockFrame16k, targetRate: CAPTURE_RATE },
  });
  // A worklet with no path to a destination is never pulled, so route its
  // (silent) output through a zero-gain sink to keep it in the render graph
  // without making the raw mic audible.
  const captureSink = captureCtx.createGain();
  captureSink.gain.value = 0;
  source.connect(captureNode);
  captureNode.connect(captureSink);
  captureSink.connect(captureCtx.destination);
  engine.captureSource = source;
  engine.captureNode = captureNode;
  engine.captureSink = captureSink;

  captureNode.port.onmessage = (event) => {
    const block = event.data as Float32Array;
    if (!block || engine.stopped) return;
    meters.onInputLevel(rms(block));
    const ws = engine.ws;
    if (ws && ws.readyState === WebSocket.OPEN) {
      if (ws.bufferedAmount > engine.maxBacklogBytes) {
        // Outbound backlog: drop this block rather than grow the queue. No
        // timestamp is recorded — the block never reached the server, so it
        // cannot desync the send→receive latency FIFO.
        meters.onOverload();
        return;
      }
      ws.send(block.buffer);
      // Record when this block left the client. The server converts blocks in
      // order, so the oldest outstanding timestamp is popped when a converted
      // block returns to yield a measured send→receive latency.
      const times = engine.sendTimes;
      times.push(performance.now());
      if (times.length > MAX_INFLIGHT_STAMPS) {
        times.shift();
      }
    }
  };

  const playbackCtx = new AudioContext();
  engine.playbackCtx = playbackCtx;
  await playbackCtx.audioWorklet.addModule(PLAYBACK_WORKLET_URL);
  if (engine.stopped) return;

  const playbackNode = new AudioWorkletNode(playbackCtx, 'rvc-playback-processor', {
    numberOfInputs: 0,
    numberOfOutputs: 1,
    outputChannelCount: [1],
    processorOptions: {
      ringSize: engine.blockFrameSr * PLAYBACK_RING_BLOCKS,
      primeSamples: engine.blockFrameSr * OUTPUT_JITTER_BLOCKS,
      // One block above the prime target is the drain high-water mark; the
      // worklet drops the oldest samples back to target once the buffer stays
      // above it long enough (see the worklet's drain policy).
      blockSamples: engine.blockFrameSr,
      inputRate: engine.modelSr,
    },
  });
  engine.playbackNode = playbackNode;
  // The worklet reports its jitter-buffer depth (ms at model rate) on a
  // throttled cadence so the UI can surface the actual buffered latency.
  playbackNode.port.onmessage = (event) => {
    const data = event.data;
    if (data && data.type === 'buffered' && typeof data.ms === 'number') {
      meters.onBufferedMs(data.ms);
    }
  };

  // Primary output path: worklet -> MediaStream -> <audio> element, whose sink is
  // switchable via setSinkId (the only output-routing API available in the Tauri
  // webview — AudioContext.setSinkId is undefined here).
  const streamDest = playbackCtx.createMediaStreamDestination();
  playbackNode.connect(streamDest);
  engine.streamDest = streamDest;

  const audioEl = new Audio();
  audioEl.autoplay = true;
  audioEl.srcObject = streamDest.stream;
  engine.audioEl = audioEl;
  const routed = await routeAudioElement(audioEl, config.outputDeviceId);
  // A failed non-default route mutes the element above; surface it so the UI can
  // block/warn instead of silently playing on the default device.
  meters.onOutputRouteError(routed ? null : (config.outputDeviceId ?? null));
  await audioEl.play().catch(() => {
    /* autoplay may need a user gesture; the graph is already primed */
  });

  // Monitor path: the SINGLE route to the local default output ("hear
  // yourself"). This is the only default-sink playback — when the selected
  // output is the system default the `<audio>` element above is muted, so this
  // gain is authoritative: monitor off (gain 0) truly silences local playback
  // (no larsen), monitor on plays exactly one copy. When the output is a
  // specific device the element feeds that device and this adds optional local
  // monitoring. Off by default to avoid feedback.
  const monitorGain = playbackCtx.createGain();
  monitorGain.gain.value = config.monitor ? 1 : 0;
  playbackNode.connect(monitorGain);
  monitorGain.connect(playbackCtx.destination);
  engine.monitorGain = monitorGain;
}

async function teardown(engine: Engine): Promise<void> {
  engine.stopped = true;
  try {
    engine.captureNode?.port.postMessage({ type: 'stop' });
  } catch {
    /* worklet already gone */
  }
  if (engine.ws && engine.ws.readyState <= WebSocket.OPEN) {
    try {
      engine.ws.close(1000, 'client stop');
    } catch {
      /* already closing */
    }
  }
  engine.ws = null;

  engine.captureSource?.disconnect();
  engine.captureNode?.disconnect();
  engine.captureSink?.disconnect();
  engine.playbackNode?.disconnect();
  engine.monitorGain?.disconnect();
  engine.streamDest?.disconnect();

  engine.stream?.getTracks().forEach((track) => {
    track.stop();
  });
  engine.stream = null;

  if (engine.audioEl) {
    engine.audioEl.pause();
    engine.audioEl.srcObject = null;
    engine.audioEl = null;
  }

  await Promise.allSettled([
    engine.captureCtx?.close() ?? Promise.resolve(),
    engine.playbackCtx?.close() ?? Promise.resolve(),
  ]);
  engine.captureCtx = null;
  engine.playbackCtx = null;
}

export interface UseRvcStreamResult {
  status: RvcStreamStatus;
  error: string | null;
  inputLevel: number;
  outputLevel: number;
  inferMs: number;
  /** Measured mouth-to-ear latency: send→receive delta + jitter-buffer depth. */
  estLatencyMs: number;
  /** Playback jitter-buffer depth in ms, reported by the worklet. */
  bufferedMs: number;
  overload: boolean;
  modelSr: number | null;
  /**
   * `deviceId` of the selected output that could not be reached (routing failed
   * or is unsupported at runtime); `null` while output is on its intended
   * target. When set, playback is muted so audio is not leaking to the default
   * sink and the UI must warn the user.
   */
  outputRouteError: string | null;
  start: (config: RvcStreamConfig) => Promise<void>;
  stop: () => void;
  setMonitor: (on: boolean) => void;
  setOutputDevice: (deviceId: string) => Promise<void>;
}

export function useRvcStream(): UseRvcStreamResult {
  const engineRef = useRef<Engine | null>(null);
  const [status, setStatus] = useState<RvcStreamStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const [inputLevel, setInputLevel] = useState(0);
  const [outputLevel, setOutputLevel] = useState(0);
  const [inferMs, setInferMs] = useState(0);
  const [estLatencyMs, setEstLatencyMs] = useState(0);
  const [bufferedMs, setBufferedMs] = useState(0);
  const [overload, setOverload] = useState(false);
  const [modelSr, setModelSr] = useState<number | null>(null);
  const [outputRouteError, setOutputRouteError] = useState<string | null>(null);

  const stop = useCallback(() => {
    const engine = engineRef.current;
    if (!engine) {
      setStatus('idle');
      return;
    }
    engineRef.current = null;
    setStatus('stopping');
    void teardown(engine).then(() => {
      setStatus('idle');
      setInputLevel(0);
      setOutputLevel(0);
      setInferMs(0);
      setEstLatencyMs(0);
      setBufferedMs(0);
      setOverload(false);
      setOutputRouteError(null);
    });
  }, []);

  const start = useCallback(async (config: RvcStreamConfig) => {
    if (engineRef.current) {
      return;
    }
    setError(null);
    setOverload(false);
    setBufferedMs(0);
    setEstLatencyMs(0);
    setOutputRouteError(null);
    setStatus('connecting');

    const engine = newEngine();
    engineRef.current = engine;

    if (!navigator.mediaDevices?.getUserMedia) {
      engineRef.current = null;
      setStatus('error');
      setError('Microphone access is unavailable in this environment.');
      return;
    }

    const blockMs = config.blockMs ?? 300;

    try {
      engine.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          channelCount: 1,
          ...(config.inputDeviceId ? { deviceId: { exact: config.inputDeviceId } } : {}),
        },
      });

      const serverUrl = useServerStore.getState().serverUrl;
      const ws = new WebSocket(toWsUrl(serverUrl));
      ws.binaryType = 'arraybuffer';
      engine.ws = ws;

      let emaInfer = 0;
      // Measured mouth-to-ear latency = EMA of the send→receive delta (a real
      // measurement replacing the old fixed formula) plus the worklet-reported
      // jitter-buffer depth. `lastBufferedMs` mirrors the latest worklet report
      // so the binary-frame handler can add it without a React read.
      let emaMeasured = 0;
      let lastBufferedMs = 0;
      const meters: GraphMeters = {
        onInputLevel: setInputLevel,
        onOverload: () => setOverload(true),
        onOutputRouteError: setOutputRouteError,
        onBufferedMs: (ms) => {
          lastBufferedMs = ms;
          setBufferedMs(ms);
        },
      };

      await new Promise<void>((resolve, reject) => {
        const timer = window.setTimeout(() => {
          reject(new Error('Timed out waiting for the server handshake.'));
        }, HANDSHAKE_TIMEOUT_MS);

        ws.onopen = () => {
          ws.send(
            JSON.stringify({
              version: STREAM_PROTOCOL_VERSION,
              profile_id: config.profileId,
              block_ms: blockMs,
              f0_up_key: config.f0UpKey,
              f0_method: config.f0Method,
              index_rate: config.indexRate,
              rms_mix_rate: config.rmsMixRate,
              protect: config.protect,
            }),
          );
        };

        ws.onmessage = (event) => {
          // First (text) frame is the handshake; everything after is binary
          // audio, handled once the graph is built (engine.modelSr set).
          if (engine.modelSr === 0 && typeof event.data === 'string') {
            let reply: Record<string, unknown>;
            try {
              reply = JSON.parse(event.data);
            } catch {
              window.clearTimeout(timer);
              reject(new Error('Malformed handshake reply from server.'));
              return;
            }
            if (reply.error || reply.ready === false) {
              window.clearTimeout(timer);
              reject(new Error(String(reply.error ?? 'Server refused the stream.')));
              return;
            }
            const sr = Number(reply.model_sr);
            if (!Number.isFinite(sr) || sr <= 0) {
              window.clearTimeout(timer);
              reject(new Error('Server did not report a model sample rate.'));
              return;
            }

            const derived = computeBlockFrames(blockMs, sr);
            engine.modelSr = sr;
            engine.blockMs = blockMs;
            engine.blockFrame16k =
              Number(reply.block_frame_16k) > 0
                ? Number(reply.block_frame_16k)
                : derived.blockFrame16k;
            engine.blockFrameSr =
              Number(reply.block_frame_sr) > 0
                ? Number(reply.block_frame_sr)
                : derived.blockFrameSr;
            engine.maxBacklogBytes = engine.blockFrame16k * 4 * MAX_SEND_BACKLOG_BLOCKS;

            buildGraph(engine, config, meters)
              .then(() => {
                window.clearTimeout(timer);
                resolve();
              })
              .catch((err) => {
                window.clearTimeout(timer);
                reject(err instanceof Error ? err : new Error(String(err)));
              });
            return;
          }

          if (engine.stopped || !(event.data instanceof ArrayBuffer)) {
            return;
          }
          const buffer = event.data;
          if (buffer.byteLength < SERVER_HEADER_BYTES) {
            return;
          }
          // Header layout is the server module docstring's `struct '<fBBH'`.
          const header = new DataView(buffer, 0, SERVER_HEADER_BYTES);
          const blockInferMs = header.getFloat32(0, true);
          const isOverloaded = header.getUint8(4) === 1;
          const isDropped = header.getUint8(5) === 1;
          const pcm = new Float32Array(buffer, SERVER_HEADER_BYTES);

          setOutputLevel(rms(pcm));
          emaInfer = emaInfer === 0 ? blockInferMs : emaInfer * 0.7 + blockInferMs * 0.3;
          setInferMs(emaInfer);

          // Measured mouth-to-ear latency. Pop the capture-send timestamp of the
          // block this output corresponds to (FIFO). When the server dropped a
          // pending input for backpressure (`dropped`), that input's timestamp
          // never yields an output, so discard one stale stamp first to keep the
          // FIFO aligned. This resync is approximate (the flag means "≥1
          // dropped") but the estimate is EMA-smoothed and the FIFO is bounded.
          const times = engine.sendTimes;
          if (isDropped && times.length > 1) {
            times.shift();
          }
          const sentAt = times.shift();
          if (sentAt !== undefined) {
            const measured = performance.now() - sentAt;
            emaMeasured = emaMeasured === 0 ? measured : emaMeasured * 0.7 + measured * 0.3;
          }
          // Total = measured processing round-trip + the reported jitter-buffer
          // depth the converted audio still has to wait before the speaker.
          setEstLatencyMs(emaMeasured + lastBufferedMs);
          setOverload(isOverloaded);

          engine.playbackNode?.port.postMessage({ pcm }, [buffer]);
        };

        ws.onerror = () => {
          window.clearTimeout(timer);
          reject(new Error('WebSocket error while connecting to the conversion stream.'));
        };

        ws.onclose = (event) => {
          if (engine.modelSr === 0) {
            window.clearTimeout(timer);
            reject(
              new Error(
                event.reason ||
                  (event.code === 1008
                    ? 'A conversion stream is already active, or the server rejected the request.'
                    : 'The conversion stream closed before it was ready.'),
              ),
            );
          } else if (!engine.stopped) {
            engineRef.current = null;
            void teardown(engine);
            setStatus('error');
            setError(event.reason || 'The conversion stream was closed by the server.');
          }
        };
      });

      // A stop() during connect/handshake nulls the ref and tears down; don't
      // flip to "running" over the top of it.
      if (engine.stopped || engineRef.current !== engine) {
        return;
      }
      setModelSr(engine.modelSr);
      setStatus('running');
    } catch (err) {
      // If the failure was a user-initiated stop(), that path already owns
      // teardown and status; surfacing an error here would clobber it.
      if (!engine.stopped) {
        engineRef.current = null;
        await teardown(engine);
        setStatus('error');
        setError(err instanceof Error ? err.message : 'Failed to start real-time conversion.');
      }
    }
  }, []);

  const setMonitor = useCallback((on: boolean) => {
    const engine = engineRef.current;
    if (engine?.monitorGain) {
      engine.monitorGain.gain.value = on ? 1 : 0;
    }
  }, []);

  const setOutputDevice = useCallback(async (deviceId: string) => {
    const el = engineRef.current?.audioEl;
    if (!el) return;
    // `deviceId === ''` means "system default"; route accordingly and clear or
    // raise the routing warning based on whether the target was reached.
    const routed = await routeAudioElement(el, deviceId || undefined);
    setOutputRouteError(routed ? null : deviceId);
  }, []);

  // Tear the session down if the component using the hook unmounts.
  useEffect(() => {
    return () => {
      const engine = engineRef.current;
      if (engine) {
        engineRef.current = null;
        void teardown(engine);
      }
    };
  }, []);

  return {
    status,
    error,
    inputLevel,
    outputLevel,
    inferMs,
    estLatencyMs,
    bufferedMs,
    overload,
    modelSr,
    outputRouteError,
    start,
    stop,
    setMonitor,
    setOutputDevice,
  };
}
