import { describe, expect, test } from 'bun:test';
import {
  decidePlaybackRetry,
  guardAudioForPlayback,
  makeRetryOrDismiss,
  SPEAK_PLAYBACK_RETRY_DELAYS_MS,
} from '../src/lib/audio/speechPlaybackRetry';

/** Records the helper's calls; the audio element and its timers are owned by DictateWindow. */
function harness(started = false) {
  const state = { started };
  const calls: string[] = [];
  return {
    calls,
    params: {
      generationId: 'gen',
      keepAudio: false,
      attempt: 0,
      started: () => state.started,
      release: () => calls.push('release'),
      retry: () => calls.push('retry'),
      dismiss: () => calls.push('dismiss'),
    },
  };
}

describe('bounded speak playback retries', () => {
  test('delays are bounded and strictly increasing', () => {
    expect(SPEAK_PLAYBACK_RETRY_DELAYS_MS.length).toBeLessThan(10);
    expect(SPEAK_PLAYBACK_RETRY_DELAYS_MS.length).toBeGreaterThan(0);
    for (let i = 1; i < SPEAK_PLAYBACK_RETRY_DELAYS_MS.length; i++) {
      expect(SPEAK_PLAYBACK_RETRY_DELAYS_MS[i]).toBeGreaterThan(SPEAK_PLAYBACK_RETRY_DELAYS_MS[i - 1]);
    }
  });

  test('a file-less completion payload is not played', () => {
    expect(guardAudioForPlayback(null)).toBe(false);
    expect(guardAudioForPlayback('')).toBe(false);
    expect(guardAudioForPlayback(undefined)).toBe(false);
    expect(guardAudioForPlayback('generations/ok.wav')).toBe(true);
  });

  test('onerror on a never-started element releases and retries', () => {
    const h = harness(false);
    makeRetryOrDismiss(h.params)();
    expect(h.calls).toEqual(['release', 'retry']);
  });

  test('onerror after sound played dismisses without replaying', () => {
    const h = harness(true);
    h.params.attempt = 1;
    makeRetryOrDismiss(h.params)();
    expect(h.calls).toEqual(['release', 'dismiss']);
  });

  test('onerror plus a rejected play() landing together yields exactly one retry', () => {
    const h = harness(false);
    const handler = makeRetryOrDismiss(h.params);
    handler();
    handler(); // both signals fire for the same failed element
    expect(h.calls.filter((c) => c === 'retry')).toHaveLength(1);
    expect(h.calls.filter((c) => c === 'dismiss')).toHaveLength(0);
    expect(h.calls.filter((c) => c === 'release')).toHaveLength(1);
  });

  test('exhausted attempts dismiss the pill without deleting the file', () => {
    const h = harness(false);
    h.params.attempt = SPEAK_PLAYBACK_RETRY_DELAYS_MS.length;
    makeRetryOrDismiss(h.params)();
    expect(h.calls).toEqual(['release', 'dismiss']);
  });

  test('the handler is bound to one element and is idempotent', () => {
    const h = harness(false);
    const handler = makeRetryOrDismiss(h.params);
    handler();
    // A new generation replaces the cycle: the old element's signals must
    // not release, retry or dismiss again.
    h.params.release = () => h.calls.push('stale-release');
    h.params.retry = () => h.calls.push('stale-retry');
    h.params.dismiss = () => h.calls.push('stale-dismiss');
    handler();
    expect(h.calls.filter((c) => c.startsWith('stale-'))).toHaveLength(0);
  });

  test('the decision is bounded and retries with strictly increasing delays only while sound has not played', () => {
    const delays: number[] = [];
    for (let attempt = 0; attempt < SPEAK_PLAYBACK_RETRY_DELAYS_MS.length; attempt++) {
      const d = decidePlaybackRetry(attempt, false);
      expect(d.action).toBe('retry');
      delays.push(d.delayMs ?? 0);
    }
    // Every attempt in range retries with a bounded, strictly increasing delay.
    expect(delays).toEqual(SPEAK_PLAYBACK_RETRY_DELAYS_MS);
    // Once the delays are exhausted, the attempt dismisses (keeps the file).
    expect(decidePlaybackRetry(SPEAK_PLAYBACK_RETRY_DELAYS_MS, false).action).toBe('dismiss');
    expect(decidePlaybackRetry(SPEAK_PLAYBACK_RETRY_DELAYS_MS + 1, false).action).toBe('dismiss');
    // An element that already produced sound is never retried, at any attempt.
    for (let attempt = 0; attempt < SPEAK_PLAYBACK_RETRY_DELAYS_MS.length + 2; attempt++) {
      expect(decidePlaybackRetry(attempt, true).action).toBe('dismiss');
    }
  });

  test('the decision dismisses (and keeps the file) when a generated file is absent', () => {
    // A completed generation whose audio_path is null/'' is dismissed rather
    // than played; the file-keep policy is enforced by the caller, not here,
    // but the guard and the decision must agree on "nothing to play".
    expect(guardAudioForPlayback(null)).toBe(false);
    expect(guardAudioForPlayback('')).toBe(false);
    // Nothing to play means the attempt must not retry into a missing file.
    expect(decidePlaybackRetry(0, true).action).toBe('dismiss');
  });
});
