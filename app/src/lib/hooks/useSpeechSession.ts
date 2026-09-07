import { emit, listen } from '@tauri-apps/api/event';
import { useEffect, useRef, useState } from 'react';
import { apiClient } from '@/lib/api/client';
import { SpeechSessionPlayer } from '@/lib/audio/speechSessionPlayer';

export function useSpeechSession(onStart: () => void) {
  const [state, setState] = useState({
    active: false,
    paused: false,
    elapsedMs: 0,
    error: null as string | null,
  });
  const player = useRef<SpeechSessionPlayer | null>(null);
  const startRef = useRef(onStart);
  startRef.current = onStart;

  useEffect(() => {
    let disposed = false;
    let currentId: string | null = null;
    const registration = listen<string>('dictate:speech-session', ({ payload }) => {
      if (disposed) return;
      let event: { session_id?: string; state?: string };
      try {
        event = JSON.parse(payload);
      } catch {
        return;
      }
      const id = event.session_id;
      if (
        !id ||
        currentId === id ||
        ['completed', 'failed', 'cancelled'].includes(event.state ?? '')
      )
        return;
      currentId = id;
      player.current?.dispose();
      startRef.current();
      setState({ active: true, paused: false, elapsedMs: 0, error: null });
      emit('dictate:show').catch(() => {});
      const next = new SpeechSessionPlayer({
        exchange: (update) => apiClient.speechPlayback(id, update),
        audio: (generationId) => new Audio(apiClient.getAudioUrl(generationId)),
        changed: ({ done, ...update }) => {
          if (!disposed && currentId === id) setState({ ...update, active: !done });
        },
      });
      player.current = next;
      next.start();
    });
    return () => {
      disposed = true;
      registration.then((unlisten) => unlisten()).catch(() => {});
      player.current?.dispose();
    };
  }, []);

  return {
    ...state,
    togglePause: () => player.current?.togglePause(),
    stop: () => player.current?.stop(),
    dismissError: () => setState((s) => ({ ...s, error: null })),
  };
}
