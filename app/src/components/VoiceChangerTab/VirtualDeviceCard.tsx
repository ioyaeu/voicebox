import { ExternalLink, Info } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { VirtualCableOS } from './useAudioDevices';

const INSTALL_LINKS: Record<VirtualCableOS, { href: string; nameKey: string }> = {
  macos: { href: 'https://existential.audio/blackhole/', nameKey: 'voiceChanger.realtime.virtual.blackhole' },
  windows: { href: 'https://vb-audio.com/Cable/', nameKey: 'voiceChanger.realtime.virtual.vbcable' },
  other: { href: 'https://existential.audio/blackhole/', nameKey: 'voiceChanger.realtime.virtual.blackhole' },
};

interface VirtualDeviceCardProps {
  os: VirtualCableOS;
}

/**
 * Setup guidance shown when no virtual audio cable is detected. Voicebox does
 * not bundle or install these drivers; the user installs one, then routes the
 * converter's output to it and selects it as the microphone in Discord/games.
 */
export function VirtualDeviceCard({ os }: VirtualDeviceCardProps) {
  const { t } = useTranslation();
  const link = INSTALL_LINKS[os];

  return (
    <div className="space-y-2 rounded-lg border border-border bg-muted/40 p-4">
      <div className="flex items-center gap-2 text-sm font-medium">
        <Info className="h-4 w-4 text-muted-foreground" />
        {t('voiceChanger.realtime.virtual.title')}
      </div>
      <p className="text-sm text-muted-foreground">
        {t('voiceChanger.realtime.virtual.description', { name: t(link.nameKey) })}
      </p>
      <ol className="list-decimal space-y-1 pl-5 text-sm text-muted-foreground">
        <li>{t('voiceChanger.realtime.virtual.step1', { name: t(link.nameKey) })}</li>
        <li>{t('voiceChanger.realtime.virtual.step2', { name: t(link.nameKey) })}</li>
        <li>{t('voiceChanger.realtime.virtual.step3', { name: t(link.nameKey) })}</li>
      </ol>
      <a
        href={link.href}
        target="_blank"
        rel="noreferrer"
        className="inline-flex items-center gap-1.5 text-sm font-medium text-accent hover:underline"
      >
        {t('voiceChanger.realtime.virtual.install', { name: t(link.nameKey) })}
        <ExternalLink className="h-3.5 w-3.5" />
      </a>
    </div>
  );
}
