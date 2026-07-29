import { useQueryClient } from '@tanstack/react-query';
import { Link } from '@tanstack/react-router';
import { AlertCircle, CheckCircle2, Download, Loader2, Mic, Play } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { apiClient } from '@/lib/api/client';
import { RVC_DEFAULTS } from '@/lib/api/constants';
import type { RvcF0Method } from '@/lib/api/types';
import { useGenerationDetail, useExportGenerationAudio } from '@/lib/hooks/useHistory';
import { useProfiles } from '@/lib/hooks/useProfiles';
import { useStartConversion } from '@/lib/hooks/useRvc';
import { useGenerationStore } from '@/stores/generationStore';
import { usePlayerStore } from '@/stores/playerStore';
import { ConversionParams } from './ConversionParams';
import { SourceFilePicker } from './SourceFilePicker';

function SectionHeading({ step, title }: { step: number; title: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="flex h-6 w-6 items-center justify-center rounded-full bg-muted text-xs font-semibold text-muted-foreground">
        {step}
      </span>
      <h2 className="text-sm font-semibold">{title}</h2>
    </div>
  );
}

export function FileConversionPanel() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const { data: profiles, isLoading: profilesLoading, isError: profilesError } = useProfiles();

  const eligibleProfiles = useMemo(
    () => (profiles ?? []).filter((p) => p.voice_type === 'rvc' && p.rvc_has_model === true),
    [profiles],
  );

  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [selectedProfileId, setSelectedProfileId] = useState<string | null>(null);
  const [f0UpKey, setF0UpKey] = useState(RVC_DEFAULTS.f0_up_key);
  const [f0Method, setF0Method] = useState<RvcF0Method>(RVC_DEFAULTS.f0_method);
  const [indexRate, setIndexRate] = useState(RVC_DEFAULTS.index_rate);
  const [rmsMixRate, setRmsMixRate] = useState(RVC_DEFAULTS.rms_mix_rate);
  const [protect, setProtect] = useState(RVC_DEFAULTS.protect);

  const [taskId, setTaskId] = useState<string | null>(null);
  const [resultLabel, setResultLabel] = useState('');
  const [convertError, setConvertError] = useState<string | null>(null);

  // Keep the profile selection valid as the eligible list loads or changes.
  useEffect(() => {
    if (eligibleProfiles.length === 0) return;
    if (!selectedProfileId || !eligibleProfiles.some((p) => p.id === selectedProfileId)) {
      setSelectedProfileId(eligibleProfiles[0].id);
    }
  }, [eligibleProfiles, selectedProfileId]);

  const startConversion = useStartConversion();
  const exportAudio = useExportGenerationAudio();
  const pendingIds = useGenerationStore((s) => s.pendingGenerationIds);
  const setAudioWithAutoPlay = usePlayerStore((s) => s.setAudioWithAutoPlay);

  const isConverting = taskId ? pendingIds.has(taskId) : false;
  const { data: detail } = useGenerationDetail(taskId ?? '');

  // The global SSE hook refetches the history *list* on completion but not the
  // per-task detail query this panel reads. Refresh it when the task leaves the
  // pending set so the completed/failed status and audio become available.
  const wasConvertingRef = useRef(false);
  useEffect(() => {
    if (wasConvertingRef.current && !isConverting && taskId) {
      queryClient.invalidateQueries({ queryKey: ['history', taskId] });
    }
    wasConvertingRef.current = isConverting;
  }, [isConverting, taskId, queryClient]);

  // While converting, SSE only refreshes the history *list* — poll the per-task
  // detail so the stage label ("loading model" → "converting") stays live
  // instead of frozen on whatever value it had when the task started.
  useEffect(() => {
    if (!isConverting || !taskId) return;
    const id = setInterval(() => {
      queryClient.invalidateQueries({ queryKey: ['history', taskId] });
    }, 1500);
    return () => clearInterval(id);
  }, [isConverting, taskId, queryClient]);

  const isBusy = startConversion.isPending || isConverting;
  const completed = !isBusy && detail?.status === 'completed';
  const failed = !isBusy && detail?.status === 'failed';

  async function handleConvert() {
    if (!sourceFile || !selectedProfileId) return;
    setConvertError(null);
    setTaskId(null);
    setResultLabel(sourceFile.name);
    try {
      const res = await startConversion.mutateAsync({
        file: sourceFile,
        profileId: selectedProfileId,
        params: {
          f0_up_key: f0UpKey,
          f0_method: f0Method,
          index_rate: indexRate,
          rms_mix_rate: rmsMixRate,
          protect,
        },
      });
      setTaskId(res.task_id);
    } catch (err) {
      setConvertError(err instanceof Error ? err.message : t('voiceChanger.convert.failed'));
    }
  }

  function handlePlay() {
    if (!taskId) return;
    setAudioWithAutoPlay(apiClient.getAudioUrl(taskId), taskId, selectedProfileId, resultLabel);
  }

  function handleDownload() {
    if (!taskId) return;
    exportAudio.mutate({ generationId: taskId, text: resultLabel || 'conversion' });
  }

  if (profilesLoading) {
    return (
      <div className="flex items-center gap-2 py-12 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" />
        {t('common.loading')}
      </div>
    );
  }

  if (profilesError) {
    return (
      <div className="flex flex-col items-center gap-3 rounded-lg border border-dashed border-destructive/40 bg-destructive/5 py-12 text-center">
        <AlertCircle className="h-8 w-8 text-destructive" />
        <div className="space-y-1">
          <p className="text-sm font-medium">{t('voiceChanger.loadError.title')}</p>
          <p className="max-w-sm text-sm text-muted-foreground">
            {t('voiceChanger.loadError.description')}
          </p>
        </div>
      </div>
    );
  }

  if (eligibleProfiles.length === 0) {
    return (
      <div className="flex flex-col items-center gap-3 rounded-lg border border-dashed border-border py-12 text-center">
        <Mic className="h-8 w-8 text-muted-foreground" />
        <div className="space-y-1">
          <p className="text-sm font-medium">{t('voiceChanger.empty.title')}</p>
          <p className="max-w-sm text-sm text-muted-foreground">
            {t('voiceChanger.empty.description')}
          </p>
        </div>
        <Button asChild variant="outline">
          <Link to="/voices">{t('voiceChanger.empty.cta')}</Link>
        </Button>
      </div>
    );
  }

  return (
    <div className="max-w-2xl space-y-8">
      {/* 1. Source audio */}
      <section className="space-y-3">
        <SectionHeading step={1} title={t('voiceChanger.source.title')} />
        <SourceFilePicker
          file={sourceFile}
          onSelect={setSourceFile}
          onClear={() => setSourceFile(null)}
          disabled={isBusy}
        />
      </section>

      {/* 2. Target voice */}
      <section className="space-y-3">
        <SectionHeading step={2} title={t('voiceChanger.profile.title')} />
        <div className="space-y-1.5">
          <Label htmlFor="rvc-profile">{t('voiceChanger.profile.label')}</Label>
          <Select
            value={selectedProfileId ?? undefined}
            onValueChange={setSelectedProfileId}
            disabled={isBusy}
          >
            <SelectTrigger id="rvc-profile">
              <SelectValue placeholder={t('voiceChanger.profile.placeholder')} />
            </SelectTrigger>
            <SelectContent>
              {eligibleProfiles.map((profile) => (
                <SelectItem key={profile.id} value={profile.id}>
                  {profile.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </section>

      {/* 3. Parameters */}
      <section className="space-y-3">
        <SectionHeading step={3} title={t('voiceChanger.params.title')} />
        <ConversionParams
          f0UpKey={f0UpKey}
          onF0UpKey={setF0UpKey}
          f0Method={f0Method}
          onF0Method={setF0Method}
          indexRate={indexRate}
          onIndexRate={setIndexRate}
          rmsMixRate={rmsMixRate}
          onRmsMixRate={setRmsMixRate}
          protect={protect}
          onProtect={setProtect}
          // Index presence is no longer exposed by the profile API (only
          // rvc_has_model), so keep the index-rate control usable; it is inert
          // server-side when the model has no feature index.
          hasIndex={true}
          disabled={isBusy}
        />
      </section>

      {/* Convert action */}
      <div className="space-y-4">
        <Button
          type="button"
          className="w-full gap-2"
          size="lg"
          disabled={!sourceFile || !selectedProfileId || isBusy}
          onClick={handleConvert}
        >
          {isBusy && <Loader2 className="h-4 w-4 animate-spin" />}
          {startConversion.isPending
            ? t('voiceChanger.convert.starting')
            : isConverting
              ? t('voiceChanger.convert.inProgress')
              : t('voiceChanger.convert.action')}
        </Button>

        {isConverting && (
          <div className="flex items-center gap-2 rounded-md border border-border bg-muted/40 px-3 py-2.5 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 shrink-0 animate-spin" />
            {detail?.status === 'loading_model'
              ? t('voiceChanger.convert.loadingModel')
              : t('voiceChanger.convert.inProgress')}
          </div>
        )}

        {convertError && (
          <div className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
            <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
            <span className="break-words">{convertError}</span>
          </div>
        )}

        {failed && (
          <div className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
            <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
            <span className="break-words">
              {detail?.error || t('voiceChanger.convert.failed')}
            </span>
          </div>
        )}

        {completed && (
          <div className="space-y-3 rounded-lg border border-border p-4">
            <div className="flex items-center gap-2 text-sm font-medium text-accent">
              <CheckCircle2 className="h-4 w-4" />
              {t('voiceChanger.result.title')}
            </div>
            <div className="flex flex-wrap gap-2">
              <Button type="button" className="gap-2" onClick={handlePlay}>
                <Play className="h-4 w-4" />
                {t('voiceChanger.result.play')}
              </Button>
              <Button
                type="button"
                variant="outline"
                className="gap-2"
                onClick={handleDownload}
                disabled={exportAudio.isPending}
              >
                <Download className="h-4 w-4" />
                {exportAudio.isPending
                  ? t('voiceChanger.result.downloading')
                  : t('voiceChanger.result.download')}
              </Button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
