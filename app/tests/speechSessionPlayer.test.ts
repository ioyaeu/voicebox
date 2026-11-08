import { describe, expect, test } from 'bun:test';
import {
  type PlaybackUpdate,
  type SpeechPlaybackStatus,
  SpeechSessionPlayer,
  type SpeechSnapshot,
} from '../src/lib/audio/speechSessionPlayer';

class FakeAudio {
  currentTime = 4;
  onended: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onplaying: (() => void) | null = null;
  onwaiting: (() => void) | null = null;
  onpause: (() => void) | null = null;
  plays = 0;
  pauses = 0;
  async play() {
    this.plays++;
  }
  pause() {
    this.pauses++;
    this.onpause?.();
  }
  removeAttribute() {}
  load() {}
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 5));
const snapshot = (sequence = 0): SpeechSnapshot => ({
  session_id: 'test',
  state: 'draining',
  ready: [{ sequence, generation_id: `gen-${sequence}` }],
  error: null,
});

describe('ordered speech playback', () => {
  test('speaking follows media playback, not an open session or pending generation', async () => {
    let server: SpeechSnapshot = { ...snapshot(), ready: [], generating: false };
    const states: { status: SpeechPlaybackStatus; elapsedMs: number; paused: boolean }[] = [];
    const audio = new FakeAudio();
    audio.currentTime = 0;
    const player = new SpeechSessionPlayer({
      exchange: async () => server,
      audio: () => audio as unknown as HTMLAudioElement,
      changed: (state) => states.push(state),
    });
    try {
      player.start();
      await settle();
      expect(states.at(-1)?.status).toBe('waiting');
      expect(audio.plays).toBe(0);
      server = { ...server, generating: true };
      player.start();
      await settle();
      expect(states.at(-1)?.status).toBe('generating');
      expect(audio.plays).toBe(0);
      server = { ...snapshot(), generating: false };
      player.start();
      await settle();
      expect(states.at(-1)?.status).toBe('generating');
      audio.onplaying!();
      expect(states.at(-1)?.status).toBe('playing');
      audio.onwaiting!();
      expect(states.at(-1)?.status).toBe('generating');
      audio.onplaying!();
      player.togglePause();
      expect(states.at(-1)?.paused).toBe(true);
      expect(states.at(-1)?.status).not.toBe('playing');
      player.togglePause();
      audio.onplaying!();
      server = { ...server, ready: [] };
      audio.currentTime = 3;
      audio.onended!();
      await settle();
      expect(states.at(-1)?.status).toBe('waiting');
      expect(states.at(-1)?.elapsedMs).toBe(3000);
      expect(audio.onplaying).toBeNull();
    } finally {
      player.dispose();
    }
  });

  test('pause does not acknowledge, and ended advances exactly once', async () => {
    const updates: PlaybackUpdate[] = [];
    const audios: FakeAudio[] = [];
    const player = new SpeechSessionPlayer({
      exchange: async (update) => {
        updates.push(update);
        return snapshot((update.acknowledged ?? -1) + 1);
      },
      audio: () => {
        const audio = new FakeAudio();
        audios.push(audio);
        return audio as unknown as HTMLAudioElement;
      },
      changed: () => {},
    });
    try {
      player.start();
      await settle();
      player.start();
      await settle();
      expect(audios.length).toBe(1);
      player.togglePause();
      player.start();
      await settle();
      expect(updates.every((update) => update.acknowledged === undefined)).toBe(true);
      expect(audios[0].pauses).toBe(1);
      player.togglePause();
      const lateEnded = audios[0].onended!;
      lateEnded();
      await settle();
      expect(audios.length).toBe(2);
      lateEnded();
      await settle();
      expect(audios.length).toBe(2);
      expect(updates.at(-1)?.acknowledged).toBe(0);
      expect(audios[0].onended).toBeNull();
    } finally {
      player.dispose();
    }
  });

  test('lost acknowledgement response never replays an already heard segment', async () => {
    const audios: FakeAudio[] = [];
    let loseReply = true;
    const player = new SpeechSessionPlayer({
      exchange: async (update) => {
        if (update.acknowledged === 0 && loseReply) {
          loseReply = false;
          throw new Error('network');
        }
        return snapshot((update.acknowledged ?? -1) + 1);
      },
      audio: () => {
        const audio = new FakeAudio();
        audios.push(audio);
        return audio as unknown as HTMLAudioElement;
      },
      changed: () => {},
    });
    try {
      player.start();
      await settle();
      audios[0].onended!();
      await settle();
      player.start();
      await settle();
      expect(audios.length).toBe(1);
      player.togglePause();
      player.start();
      await settle();
      expect(audios.length).toBe(2);
      expect(audios[0].plays).toBe(1);
    } finally {
      player.dispose();
    }
  });

  test('stop during a slow request cannot start the returned audio', async () => {
    let reply!: (value: SpeechSnapshot) => void;
    let played = false;
    const player = new SpeechSessionPlayer({
      exchange: () =>
        new Promise((resolve) => {
          reply = resolve;
        }),
      audio: () => {
        played = true;
        return new FakeAudio() as unknown as HTMLAudioElement;
      },
      changed: () => {},
    });
    try {
      player.start();
      player.stop();
      reply(snapshot());
      await settle();
      expect(played).toBe(false);
    } finally {
      player.dispose();
    }
  });

  test('failed audio reports failure without acknowledging completion', async () => {
    const updates: PlaybackUpdate[] = [];
    const audio = new FakeAudio();
    const player = new SpeechSessionPlayer({
      exchange: async (update) => {
        updates.push(update);
        return update.error ? { ...snapshot(), state: 'failed', error: update.error } : snapshot();
      },
      audio: () => audio as unknown as HTMLAudioElement,
      changed: () => {},
    });
    try {
      player.start();
      await settle();
      audio.onerror!();
      await settle();
      expect(updates.at(-1)?.error).toBe('Audio playback failed');
      expect(updates.at(-1)?.acknowledged).toBeUndefined();
    } finally {
      player.dispose();
    }
  });
});
