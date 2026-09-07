export interface SpeechSnapshot {
  session_id: string;
  state: 'accepting' | 'draining' | 'completed' | 'cancelled' | 'failed';
  ready: { sequence: number; generation_id: string }[];
  error: string | null;
}

export interface PlaybackUpdate {
  renderer: string;
  acknowledged?: number;
  error?: string;
  stopped?: boolean;
}

interface PlayerIO {
  exchange: (update: PlaybackUpdate) => Promise<SpeechSnapshot>;
  audio: (generationId: string) => HTMLAudioElement;
  changed: (state: {
    paused: boolean;
    elapsedMs: number;
    error: string | null;
    done: boolean;
  }) => void;
}

/** One renderer, one audio element, one in-flight acknowledgement. A failed
 * request retries the acknowledgement, never the already-heard audio.
 */
export class SpeechSessionPlayer {
  private readonly renderer = crypto.randomUUID();
  private audio: HTMLAudioElement | null = null;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private disposed = false;
  private busy = false;
  private paused = false;
  private elapsed = 0;
  private acknowledged: number | undefined;
  private error: string | undefined;
  private stopped = false;
  private failures = 0;

  constructor(private readonly io: PlayerIO) {}

  start() {
    void this.tick();
  }

  togglePause() {
    this.paused = !this.paused;
    if (this.paused) this.audio?.pause();
    else if (this.audio) void this.audio.play().catch((error) => this.fail(String(error)));
    this.report();
  }

  stop() {
    this.stopped = true;
    this.releaseAudio();
    void this.tick();
  }

  dispose() {
    this.disposed = true;
    clearTimeout(this.timer);
    this.releaseAudio();
  }

  private report(done = false) {
    this.io.changed({
      paused: this.paused,
      elapsedMs: this.elapsed + (this.audio?.currentTime ?? 0) * 1000,
      error: this.error ?? null,
      done,
    });
  }

  private releaseAudio() {
    if (!this.audio) return;
    this.audio.onended = null;
    this.audio.onerror = null;
    this.audio.pause();
    this.audio.removeAttribute('src');
    this.audio.load();
    this.audio = null;
  }

  private fail(message: string) {
    if (this.disposed) return;
    this.error = message.slice(0, 300);
    this.releaseAudio();
    this.report();
    void this.tick();
  }

  private async tick() {
    if (this.disposed || this.busy) return;
    clearTimeout(this.timer);
    this.busy = true;
    try {
      const snapshot = await this.io.exchange({
        renderer: this.renderer,
        acknowledged: this.acknowledged,
        error: this.error,
        stopped: this.stopped,
      });
      if (this.disposed) return;
      this.failures = 0;
      if (['completed', 'cancelled', 'failed'].includes(snapshot.state)) {
        this.error = snapshot.error ?? this.error;
        this.dispose();
        this.report(true);
        return;
      }
      // stop/pause may have happened while exchange was awaiting its reply.
      const next = snapshot.ready[0];
      if (
        !this.stopped &&
        !this.error &&
        !this.paused &&
        !this.audio &&
        next &&
        (this.acknowledged === undefined || next.sequence > this.acknowledged)
      ) {
        const audio = this.io.audio(next.generation_id);
        this.audio = audio;
        audio.onended = () => {
          if (this.audio !== audio || this.disposed) return;
          this.elapsed += audio.currentTime * 1000;
          this.acknowledged = next.sequence;
          this.releaseAudio();
          void this.tick();
        };
        audio.onerror = () => {
          if (this.audio === audio) this.fail('Audio playback failed');
        };
        void audio.play().catch((error) => {
          if (this.audio === audio) this.fail(String(error));
        });
      }
      this.report();
    } catch (error) {
      // Never keep playing disconnected from the authoritative session state.
      this.audio?.pause();
      this.paused = true;
      this.failures += 1;
      if (this.failures >= 5) {
        this.error = String(error);
        this.dispose();
        this.report(true);
      } else this.report();
    } finally {
      this.busy = false;
      if (!this.disposed) this.timer = setTimeout(() => void this.tick(), 1000);
    }
  }
}
