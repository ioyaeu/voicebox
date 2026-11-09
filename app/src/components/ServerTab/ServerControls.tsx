import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Loader2, Play, RotateCw, Square } from 'lucide-react';
import type { ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import { create } from 'zustand';
import { Button } from '@/components/ui/button';
import { useToast } from '@/components/ui/use-toast';
import { usePlatform } from '@/platform/PlatformContext';
import { isLoopbackVoiceboxServerUrl, useServerStore } from '@/stores/serverStore';

type ServerAction = 'start' | 'stop' | 'restart';

/**
 * The server action in progress. Kept outside the component so that leaving
 * the settings page mid-action and coming back still shows it in progress,
 * instead of offering another action against a server that is still booting.
 */
const useServerActionStore = create<{
  pending: ServerAction | null;
  setPending: (pending: ServerAction | null) => void;
}>()((set) => ({
  pending: null,
  setPending: (pending) => set({ pending }),
}));

const PENDING_LABEL = {
  start: 'settings.general.server.starting',
  stop: 'settings.general.server.stopping',
  restart: 'settings.general.server.restarting',
} as const;

const DONE_TITLE = {
  start: 'settings.general.server.started',
  stop: 'settings.general.server.stopped',
  restart: 'settings.general.server.restarted',
} as const;

const FAILED_TITLE = {
  start: 'settings.general.server.startFailed',
  stop: 'settings.general.server.stopFailed',
  restart: 'settings.general.server.restartFailed',
} as const;

const BUTTON_CLASS = 'h-7 px-2.5 text-xs [&_svg]:size-3.5';

/**
 * The server status pill, plus start / stop / restart buttons when this app
 * controls the server process: packaged desktop app, local server URL, and a
 * server the app spawned or adopted. In dev mode, in the web app, or against an
 * external or remote server, only the status is shown.
 */
export function ServerControls({
  status,
  online,
}: {
  /** The connection status pill, shown when no action is in progress. */
  status: ReactNode;
  /** Health check result: true online, false offline, undefined still checking. */
  online: boolean | undefined;
}) {
  const { t } = useTranslation();
  const platform = usePlatform();
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const serverUrl = useServerStore((state) => state.serverUrl);
  const pending = useServerActionStore((state) => state.pending);
  const setPending = useServerActionStore((state) => state.setPending);

  const controllable =
    platform.metadata.isTauri &&
    Boolean(import.meta.env?.PROD) &&
    isLoopbackVoiceboxServerUrl(serverUrl);

  // Re-checked whenever the health status flips, since stopping, starting or a
  // crash is what changes it.
  const { data: managed } = useQuery({
    queryKey: ['server', 'managed', online],
    queryFn: () => platform.lifecycle.isServerManaged(),
    enabled: controllable,
  });

  if (!controllable) {
    return <>{status}</>;
  }

  async function run(action: ServerAction) {
    if (useServerActionStore.getState().pending) {
      return;
    }
    setPending(action);
    try {
      if (action === 'stop') {
        await platform.lifecycle.stopServer();
        window.__voiceboxServerStartedByApp = false;
      } else {
        // Both apply the current network-access setting. Start never stops
        // anything: if this app's server turns out to be alive (the health
        // check failed for another reason), it is left running.
        const remote = useServerStore.getState().mode === 'remote';
        const url =
          action === 'start'
            ? await platform.lifecycle.startServer(remote)
            : await platform.lifecycle.restartServer(undefined, remote);
        window.__voiceboxServerStartedByApp = true;
        useServerStore.getState().setServerUrl(url);
      }
      toast({ title: t(DONE_TITLE[action]) });
    } catch (error) {
      toast({
        title: t(FAILED_TITLE[action]),
        description: error instanceof Error ? error.message : String(error),
        variant: 'destructive',
      });
    } finally {
      // Refresh the server status before leaving the pending state, so the pill
      // never shows the pre-action status next to the post-action controls
      // (e.g. "Online" + "External" right after a stop).
      if (action === 'stop') {
        // Drop the cached "online" result rather than keep it while re-checking.
        await queryClient.resetQueries({ queryKey: ['server'] });
      } else {
        // A (re)started server also reset everything the backend serves
        // (loaded models, etc.).
        void queryClient.invalidateQueries({
          predicate: (query) => query.queryKey[0] !== 'server',
        });
        await queryClient.invalidateQueries({ queryKey: ['server'] });
      }
      setPending(null);
    }
  }

  if (pending) {
    return (
      <div className="flex items-center gap-2 rounded-full border border-border/60 px-3 py-1">
        <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />
        <span className="text-xs text-muted-foreground">{t(PENDING_LABEL[pending])}</span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-1.5">
      {status}
      {online === true && managed === true && (
        <>
          <Button
            variant="outline"
            size="sm"
            className={BUTTON_CLASS}
            title={t('settings.general.server.restartHint')}
            onClick={() => run('restart')}
          >
            <RotateCw />
            {t('settings.general.server.restart')}
          </Button>
          <Button
            variant="outline"
            size="sm"
            className={BUTTON_CLASS}
            title={t('settings.general.server.stopHint')}
            onClick={() => run('stop')}
          >
            <Square />
            {t('settings.general.server.stop')}
          </Button>
        </>
      )}
      {online === true && managed === false && (
        <span
          className="text-xs text-muted-foreground"
          title={t('settings.general.server.externalHint')}
        >
          {t('settings.general.server.external')}
        </span>
      )}
      {online === false && (
        <Button
          variant="outline"
          size="sm"
          className={BUTTON_CLASS}
          title={t('settings.general.server.startHint')}
          onClick={() => run('start')}
        >
          <Play />
          {t('settings.general.server.start')}
        </Button>
      )}
    </div>
  );
}
