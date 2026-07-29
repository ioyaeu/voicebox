import { AlertCircle, AlertTriangle, Gauge, Headphones, Loader2, Mic, Radio, Square } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { AudioBars, type AudioBarsMode } from '@/components/AudioBars';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Toggle } from '@/components/ui/toggle';
import { RVC_DEFAULTS } from '@/lib/api/constants';
import type { RvcF0Method } from '@/lib/api/types';
import { useProfiles } from '@/lib/hooks/useProfiles';
import { useRvcStream } from '@/lib/hooks/useRvcStream';
import { useUIStore } from '@/stores/uiStore';
import { Link } from '@tanstack/react-router';
import { ConversionParams } from './ConversionParams';
import { DEFAULT_DEVICE_VALUE, DeviceSelect } from './DeviceSelect';
import { useAudioDevices } from './useAudioDevices';
import { VirtualDeviceCard } from './VirtualDeviceCard';

// Above this level the meter reads "live"; below it the bars settle to idle.
const LEVEL_ACTIVE_THRESHOLD = 0.015;
// Rough mouth-to-ear budget from 05_REALTIME_STREAMING.md; over it the gauge warns.
const LATENCY_WARN_MS = 700;

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

function meterMode(level: number, active: boolean): AudioBarsMode {
  if (!active) return 'idle';
  return level > LEVEL_ACTIVE_THRESHOLD ? 'playing' : 'idle';
}

export function RealtimeConversionPanel() {
  const { t } = useTranslation();
  const { data: profiles, isLoading: profilesLoading } = useProfiles();
  const devices = useAudioDevices();
  const stream = useRvcStream();

  const eligibleProfiles = useMemo(
    () => (profiles ?? []).filter((p) => p.voice_type === 'rvc' && p.rvc_has_model === true),
    [profiles],
  );

  const [selectedProfileId, setSelectedProfileId] = useState<string | null>(null);
  const [f0UpKey, setF0UpKey] = useState(RVC_DEFAULTS.f0_up_key);
  const [f0Method, setF0Method] = useState<RvcF0Method>(RVC_DEFAULTS.f0_method);
  const [indexRate, setIndexRate] = useState(RVC_DEFAULTS.index_rate);
  const [rmsMixRate, setRmsMixRate] = useState(RVC_DEFAULTS.rms_mix_rate);
  const [protect, setProtect] = useState(RVC_DEFAULTS.protect);
  const [inputDeviceValue, setInputDeviceValue] = useState(DEFAULT_DEVICE_VALUE);
  // Restore the persisted output sink so a returning user keeps e.g. BlackHole
  // selected before device enumeration completes on a fresh launch. `null` in the
  // store means "system default"; the panel represents that with the sentinel.
  const setPersistedOutputSink = useUIStore((s) => s.setRealtimeOutputDeviceId);
  const [outputDeviceValue, setOutputDeviceValue] = useState(
    () => useUIStore.getState().realtimeOutputDeviceId ?? DEFAULT_DEVICE_VALUE,
  );
  const [monitor, setMonitorState] = useState(false);
  const [requestingPermission, setRequestingPermission] = useState(false);

  useEffect(() => {
    if (eligibleProfiles.length === 0) return;
    if (!selectedProfileId || !eligibleProfiles.some((p) => p.id === selectedProfileId)) {
      setSelectedProfileId(eligibleProfiles[0].id);
    }
  }, [eligibleProfiles, selectedProfileId]);

  const isRunning = stream.status === 'running';
  const isBusy = stream.status === 'connecting' || stream.status === 'stopping';
  const { setMonitor: applyMonitor, setOutputDevice: applyOutputDevice } = stream;

  // Larsen (feedback) risk: converted audio reaches local speakers while the mic
  // is open. That happens whenever monitoring (local fan-out to the default
  // sink), or whenever the running stream plays to a real, non-virtual output
  // device the room can hear — not only when the monitor toggle is on. A virtual
  // cable (e.g. BlackHole) is not audible locally, so it carries no such risk.
  const outputIsVirtual =
    !!devices.virtualOutput && outputDeviceValue === devices.virtualOutput.deviceId;
  const outputIsDefault = outputDeviceValue === DEFAULT_DEVICE_VALUE;
  const feedbackRisk = monitor || (isRunning && !outputIsDefault && !outputIsVirtual);

  // A persisted output sink can point at a device that is gone (unplugged) or
  // simply not enumerated yet. Once labels are available we can tell: warn
  // (never clear the selection) when the selected non-default sink is absent, so
  // the user learns why their saved device is not receiving audio.
  const persistedSinkMissing =
    devices.outputRoutingSupported &&
    devices.labelsAvailable &&
    outputDeviceValue !== DEFAULT_DEVICE_VALUE &&
    !devices.outputs.some((o) => o.deviceId === outputDeviceValue);

  async function handleStart() {
    if (!selectedProfileId) return;
    await stream.start({
      profileId: selectedProfileId,
      f0UpKey,
      f0Method,
      indexRate,
      rmsMixRate,
      protect,
      inputDeviceId:
        inputDeviceValue === DEFAULT_DEVICE_VALUE ? undefined : inputDeviceValue,
      outputDeviceId:
        outputDeviceValue === DEFAULT_DEVICE_VALUE ? undefined : outputDeviceValue,
      monitor,
    });
    // Labels are only populated once mic permission is granted.
    void devices.refresh();
  }

  function handleOutputChange(value: string) {
    setOutputDeviceValue(value);
    // Persist the choice (null = system default) so it survives a relaunch.
    setPersistedOutputSink(value === DEFAULT_DEVICE_VALUE ? null : value);
    if (isRunning) {
      // Empty string resets an <audio> element to the system default sink.
      void applyOutputDevice(value === DEFAULT_DEVICE_VALUE ? '' : value);
    }
  }

  async function handleGrantPermission() {
    setRequestingPermission(true);
    try {
      // Preflight the mic permission so enumerateDevices() returns labels; this
      // is what makes BlackHole (and every named device) appear on a fresh
      // launch without starting a conversion.
      await devices.requestPermission();
    } finally {
      setRequestingPermission(false);
    }
  }

  function handleMonitorChange(value: boolean) {
    setMonitorState(value);
    applyMonitor(value);
  }

  if (profilesLoading) {
    return (
      <div className="flex items-center gap-2 py-12 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" />
        {t('common.loading')}
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

  const paramsDisabled = isRunning || isBusy;

  return (
    <div className="max-w-2xl space-y-8">
      {/* 1. Target voice */}
      <section className="space-y-3">
        <SectionHeading step={1} title={t('voiceChanger.profile.title')} />
        <div className="space-y-1.5">
          <Label htmlFor="rvc-rt-profile">{t('voiceChanger.profile.label')}</Label>
          <Select
            value={selectedProfileId ?? undefined}
            onValueChange={setSelectedProfileId}
            disabled={paramsDisabled}
          >
            <SelectTrigger id="rvc-rt-profile">
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

      {/* 2. Devices */}
      <section className="space-y-3">
        <SectionHeading step={2} title={t('voiceChanger.realtime.devices.title')} />
        <div className="grid gap-4 sm:grid-cols-2">
          <DeviceSelect
            id="rvc-rt-input"
            label={t('voiceChanger.realtime.devices.input')}
            defaultOptionLabel={t('voiceChanger.realtime.devices.systemDefault')}
            devices={devices.inputs}
            value={inputDeviceValue}
            onValueChange={setInputDeviceValue}
            disabled={isRunning || isBusy}
            placeholder={t('voiceChanger.realtime.devices.systemDefault')}
          />
          {devices.outputRoutingSupported ? (
            <DeviceSelect
              id="rvc-rt-output"
              label={t('voiceChanger.realtime.devices.output')}
              defaultOptionLabel={t('voiceChanger.realtime.devices.systemDefault')}
              devices={devices.outputs}
              value={outputDeviceValue}
              onValueChange={handleOutputChange}
              disabled={isBusy}
              placeholder={t('voiceChanger.realtime.devices.systemDefault')}
            />
          ) : (
            <div className="flex items-start gap-2 rounded-md border border-border bg-muted/40 p-3 text-sm text-muted-foreground">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
              <span>{t('voiceChanger.realtime.devices.outputUnsupported')}</span>
            </div>
          )}
        </div>
        {!devices.labelsAvailable && (
          <div className="flex flex-col items-start gap-2 rounded-md border border-border bg-muted/40 p-3">
            <p className="text-xs text-muted-foreground">
              {t('voiceChanger.realtime.devices.permissionHint')}
            </p>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="gap-2"
              onClick={handleGrantPermission}
              disabled={requestingPermission}
            >
              {requestingPermission ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Mic className="h-4 w-4" />
              )}
              {t('voiceChanger.realtime.devices.grantAccess')}
            </Button>
          </div>
        )}
        {persistedSinkMissing && (
          <div className="flex items-start gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-600 dark:text-amber-400">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <span className="break-words">
              {t('voiceChanger.realtime.devices.persistedSinkMissing')}
            </span>
          </div>
        )}
        {isRunning && stream.outputRouteError && (
          <div className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <span className="break-words">
              {t('voiceChanger.realtime.devices.outputRouteFailed', {
                device:
                  devices.outputs.find((o) => o.deviceId === stream.outputRouteError)?.label ??
                  stream.outputRouteError,
              })}
            </span>
          </div>
        )}
        {!devices.hasVirtualOutput && <VirtualDeviceCard os={devices.os} />}
      </section>

      {/* 3. Monitor */}
      <section className="space-y-3">
        <SectionHeading step={3} title={t('voiceChanger.realtime.monitor.title')} />
        <div className="flex items-center justify-between rounded-md border border-border p-3">
          <div className="flex items-center gap-2">
            <Headphones className="h-4 w-4 text-muted-foreground" />
            <Label htmlFor="rvc-rt-monitor" className="cursor-pointer">
              {t('voiceChanger.realtime.monitor.label')}
            </Label>
          </div>
          <Toggle id="rvc-rt-monitor" checked={monitor} onCheckedChange={handleMonitorChange} />
        </div>
        {feedbackRisk ? (
          <div className="flex items-start gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-600 dark:text-amber-400">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>{t('voiceChanger.realtime.monitor.warning')}</span>
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">
            {t('voiceChanger.realtime.monitor.hint')}
          </p>
        )}
      </section>

      {/* 4. Parameters */}
      <section className="space-y-3">
        <SectionHeading step={4} title={t('voiceChanger.params.title')} />
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
          disabled={paramsDisabled}
        />
      </section>

      {/* Live meters + latency */}
      <section className="grid gap-3 sm:grid-cols-3">
        <div className="flex items-center justify-between rounded-md border border-border p-3">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Mic className="h-4 w-4" />
            {t('voiceChanger.realtime.meters.input')}
          </div>
          <AudioBars mode={meterMode(stream.inputLevel, isRunning)} />
        </div>
        <div className="flex items-center justify-between rounded-md border border-border p-3">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Radio className="h-4 w-4" />
            {t('voiceChanger.realtime.meters.output')}
          </div>
          <AudioBars mode={meterMode(stream.outputLevel, isRunning)} />
        </div>
        <div
          className={`flex items-center justify-between rounded-md border p-3 ${
            isRunning && stream.estLatencyMs > LATENCY_WARN_MS
              ? 'border-amber-500/40 bg-amber-500/10'
              : 'border-border'
          }`}
        >
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Gauge className="h-4 w-4" />
            {t('voiceChanger.realtime.meters.latency')}
          </div>
          <div className="flex flex-col items-end">
            <span className="text-sm font-medium tabular-nums">
              {isRunning
                ? t('voiceChanger.realtime.meters.latencyValue', {
                    ms: Math.round(stream.estLatencyMs),
                  })
                : '—'}
            </span>
            {isRunning && (
              <span className="text-[10px] text-muted-foreground tabular-nums">
                {t('voiceChanger.realtime.meters.bufferedValue', {
                  ms: Math.round(stream.bufferedMs),
                })}
              </span>
            )}
          </div>
        </div>
      </section>

      {isRunning && stream.overload && (
        <div className="flex items-start gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-600 dark:text-amber-400">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
          <span>
            {t('voiceChanger.realtime.overload', { ms: Math.round(stream.inferMs) })}
          </span>
        </div>
      )}

      {/* Start / stop */}
      <div className="space-y-4">
        {isRunning ? (
          <Button type="button" variant="destructive" className="w-full gap-2" size="lg" onClick={stream.stop}>
            <Square className="h-4 w-4" />
            {t('voiceChanger.realtime.stop')}
          </Button>
        ) : (
          <Button
            type="button"
            className="w-full gap-2"
            size="lg"
            disabled={!selectedProfileId || isBusy}
            onClick={handleStart}
          >
            {isBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Radio className="h-4 w-4" />}
            {stream.status === 'connecting'
              ? t('voiceChanger.realtime.connecting')
              : t('voiceChanger.realtime.start')}
          </Button>
        )}

        {stream.error && (
          <div className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <span className="break-words">{stream.error}</span>
          </div>
        )}
      </div>
    </div>
  );
}
