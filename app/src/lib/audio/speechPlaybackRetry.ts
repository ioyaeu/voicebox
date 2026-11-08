/**
 * Bounded playback retries for the agent-speak pill cycle.
 *
 * A failed playback attempt (`onerror`) or a rejected `play()` promise must
 * neither delete the temporary audio file nor start a second audio element.
 * `retryOrDismiss` releases the current element, schedules a bounded retry on
 * an element that never started, and dismisses (keeping the file) once the
 * attempts are exhausted or the sound already played.
 */

export const SPEAK_PLAYBACK_RETRY_DELAYS_MS = [250, 750, 1500];

/** True when the completion payload has a file to play (absent = failed). */
export function guardAudioForPlayback(audioPath: string | null | undefined): boolean {
  return typeof audioPath === 'string' && audioPath !== '';
}

/**
 * Pure retry/dismiss decision for one failed playback attempt.
 *
 * An element that never produced sound (`started === false`) is retried on a
 * fresh element while attempts remain; an element that did play, or an attempt
 * that exhausted the delays, is dismissed. Either way the temporary audio file
 * is kept — startup/expiry cleanup is the fallback for stale agent speech.
 * Kept as a pure function so the bounded-retry behavior is testable without a
 * DOM.
 */
export function decidePlaybackRetry(
  attempt: number,
  started: boolean,
): { action: 'retry' | 'dismiss'; delayMs?: number } {
  if (!started && attempt < SPEAK_PLAYBACK_RETRY_DELAYS_MS.length) {
    return { action: 'retry', delayMs: SPEAK_PLAYBACK_RETRY_DELAYS_MS[attempt] };
  }
  return { action: 'dismiss' };
}

/**
 * Handler shared by one audio element's `onerror` and its `play()` rejection.
 *
 * Both signals can fire for the same failed element (a missing or unreadable
 * file triggers `onerror` and rejects `play()`); the per-element flag makes
 * the handler idempotent so the element is released once and at most one retry
 * is scheduled. `started` (set by `onplaying`) decides the outcome: an element
 * that produced no sound is retried on a fresh element; one that did is
 * dismissed without replaying it. Either way the temporary file is kept —
 * startup/expiry cleanup is the fallback for stale agent speech.
 */
export function makeRetryOrDismiss(params: {
  generationId: string;
  keepAudio: boolean;
  attempt: number;
  started: () => boolean;
  release: () => void;
  retry: () => void;
  dismiss: () => void;
}): () => void {
  let handled = false;
  return () => {
    if (handled) return;
    handled = true;
    params.release();
    if (decidePlaybackRetry(params.attempt, params.started()).action === 'retry') {
      params.retry();
      return;
    }
    // A failed playback attempt must not delete temporary audio: keeping the
    // file lets the user retry/export/debug, and startup/expiry cleanup is the
    // fallback for stale agent speech.
    params.dismiss();
  };
}
