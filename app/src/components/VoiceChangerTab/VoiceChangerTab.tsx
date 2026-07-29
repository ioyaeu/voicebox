import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { FileConversionPanel } from './FileConversionPanel';
import { RealtimeConversionPanel } from './RealtimeConversionPanel';

export function VoiceChangerTab() {
  const { t } = useTranslation();
  const [mode, setMode] = useState<'file' | 'realtime'>('file');

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <header className="shrink-0 pt-2 pb-4">
        <h1 className="text-xl font-semibold">{t('voiceChanger.title')}</h1>
        <p className="text-sm text-muted-foreground">{t('voiceChanger.description')}</p>
      </header>

      <Tabs
        value={mode}
        onValueChange={(value) => setMode(value as 'file' | 'realtime')}
        className="flex min-h-0 flex-1 flex-col"
      >
        <TabsList className="shrink-0 self-start">
          <TabsTrigger value="file">{t('voiceChanger.mode.file')}</TabsTrigger>
          <TabsTrigger value="realtime">{t('voiceChanger.mode.realtime')}</TabsTrigger>
        </TabsList>

        <TabsContent value="file" className="min-h-0 flex-1 overflow-y-auto pb-8">
          <FileConversionPanel />
        </TabsContent>
        <TabsContent value="realtime" className="min-h-0 flex-1 overflow-y-auto pb-8">
          <RealtimeConversionPanel />
        </TabsContent>
      </Tabs>
    </div>
  );
}
