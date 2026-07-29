import { useQuery } from '@tanstack/react-query';
import {
  AlertCircle,
  CheckCircle2,
  FileMusic,
  Loader2,
  Replace,
  Trash2,
  Upload,
  X,
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ConversionParams } from '@/components/VoiceChangerTab/ConversionParams';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import { DEFAULT_RVC_BASE_VOICE } from '@/lib/api/constants';
import type { PresetVoice, RvcConvertParams, VoiceProfileResponse } from '@/lib/api/types';
import { useProfiles } from '@/lib/hooks/useProfiles';
import { useDeleteRvcModel, useUploadRvcModel } from '@/lib/hooks/useRvc';

const MODEL_EXTENSION = '.pth';
const INDEX_EXTENSION = '.index';

/** Cloning-free preset engines usable as the base TTS voice for the chain. */
const RVC_BASE_ENGINES = [
  { value: 'kokoro', label: 'Kokoro 82M' },
  { value: 'qwen_custom_voice', label: 'Qwen CustomVoice' },
  { value: 'voxtral', label: 'Voxtral 4B TTS' },
] as const;

/**
 * Sentinel "source" value selecting a Voicebox profile (instead of a preset
 * engine) as the chain base. Mirrors the backend `RVC_PROFILE_BASE_PREFIX`
 * ("profile:") so a profile base is stored as `"profile:{profile_id}"`. No
 * preset engine is named "profile", so the two grammars never collide.
 */
const RVC_PROFILE_SOURCE = 'profile';

/**
 * Split a stored base voice into its source + identifier. Handles both chain
 * grammars: `"{engine}:{voice_id}"` (preset base) and
 * `"profile:{profile_id}"` (profile base). Falls back to the default preset
 * when the value is missing or has no colon.
 */
export function parseRvcBaseVoice(raw: string | null | undefined): [string, string] {
  if (raw?.includes(':')) {
    const idx = raw.indexOf(':');
    return [raw.slice(0, idx), raw.slice(idx + 1)];
  }
  const [engine, voice] = DEFAULT_RVC_BASE_VOICE.split(':');
  return [engine, voice];
}

function hasExtension(file: File, extension: string): boolean {
  return file.name.toLowerCase().endsWith(extension);
}

/** Inline, user-actionable error surface for the 400 validator messages. */
function RvcErrorBox({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
      <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
      <span className="break-words">{message}</span>
    </div>
  );
}

function RvcUploadProgress({ fraction }: { fraction: number }) {
  const { t } = useTranslation();
  const percent = Math.min(100, Math.round(fraction * 100));
  // Once the body is fully sent (fraction 1) the request is still open while the
  // server validates the checkpoint — surface that instead of a stuck "100%".
  const validating = fraction >= 1;
  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" />
        {validating
          ? t('profileForm.rvc.validating')
          : t('profileForm.rvc.uploading', { percent })}
      </div>
      <Progress value={percent} className="h-2" />
    </div>
  );
}

interface RvcFileFieldProps {
  label: string;
  hint: string;
  chooseLabel: string;
  accept: string;
  file: File | null;
  onSelect: (file: File) => void;
  onRemove: () => void;
  disabled?: boolean;
}

function RvcFileField({
  label,
  hint,
  chooseLabel,
  accept,
  file,
  onSelect,
  onRemove,
  disabled = false,
}: RvcFileFieldProps) {
  const { t } = useTranslation();
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <div className="space-y-1.5">
      <div className="text-sm font-medium">{label}</div>
      <p className="text-xs text-muted-foreground">{hint}</p>
      <input
        type="file"
        accept={accept}
        ref={inputRef}
        disabled={disabled}
        onChange={(e) => {
          const selected = e.target.files?.[0];
          if (selected) {
            onSelect(selected);
          }
          // Allow re-selecting the same filename after a validation reject.
          e.target.value = '';
        }}
        className="hidden"
      />
      {file ? (
        <div className="flex items-center gap-2 rounded-md border border-border bg-muted/40 px-3 py-2">
          <FileMusic className="h-4 w-4 shrink-0 text-accent" />
          <span className="flex-1 truncate text-sm">{file.name}</span>
          <Button
            type="button"
            size="icon"
            variant="ghost"
            className="h-7 w-7"
            disabled={disabled}
            onClick={onRemove}
            aria-label={t('profileForm.rvc.remove')}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      ) : (
        <Button
          type="button"
          variant="outline"
          className="w-full justify-start gap-2"
          disabled={disabled}
          onClick={() => inputRef.current?.click()}
        >
          <Upload className="h-4 w-4" />
          {chooseLabel}
        </Button>
      )}
    </div>
  );
}

interface RvcModelUploadFieldsProps {
  modelFile: File | null;
  indexFile: File | null;
  onModelFile: (file: File | null) => void;
  onIndexFile: (file: File | null) => void;
  /** Called with a validation message (or null to clear) on a rejected file. */
  onError: (message: string | null) => void;
  /** Upload fraction (0..1) while a submit is in flight, else null. */
  fraction: number | null;
  error: string | null;
  disabled?: boolean;
}

/**
 * The paired model (`.pth`) + index (`.index`) pickers plus their shared
 * progress/error surface. Extracted so the create picker and the edit-mode
 * upload/replace flows render one identical control instead of three copies.
 */
function RvcModelUploadFields({
  modelFile,
  indexFile,
  onModelFile,
  onIndexFile,
  onError,
  fraction,
  error,
  disabled = false,
}: RvcModelUploadFieldsProps) {
  const { t } = useTranslation();
  return (
    <>
      <RvcFileField
        label={t('profileForm.rvc.modelLabel')}
        hint={t('profileForm.rvc.modelHint')}
        chooseLabel={t('profileForm.rvc.chooseModel')}
        accept={MODEL_EXTENSION}
        file={modelFile}
        disabled={disabled}
        onSelect={(file) => {
          if (!hasExtension(file, MODEL_EXTENSION)) {
            onError(t('profileForm.rvc.validation.modelExtension'));
            return;
          }
          onError(null);
          onModelFile(file);
        }}
        onRemove={() => onModelFile(null)}
      />
      <RvcFileField
        label={t('profileForm.rvc.indexLabel')}
        hint={t('profileForm.rvc.indexHint')}
        chooseLabel={t('profileForm.rvc.chooseIndex')}
        accept={INDEX_EXTENSION}
        file={indexFile}
        disabled={disabled}
        onSelect={(file) => {
          if (!hasExtension(file, INDEX_EXTENSION)) {
            onError(t('profileForm.rvc.validation.indexExtension'));
            return;
          }
          onError(null);
          onIndexFile(file);
        }}
        onRemove={() => onIndexFile(null)}
      />
      {fraction != null && <RvcUploadProgress fraction={fraction} />}
      {error && <RvcErrorBox message={error} />}
    </>
  );
}

/** Read-only summary of a validated checkpoint's metadata. */
function RvcMetadata({ profile }: { profile: VoiceProfileResponse }) {
  const { t } = useTranslation();
  return (
    <div className="rounded-lg border border-border p-3 space-y-2">
      <div className="text-sm font-medium">{t('profileForm.rvc.metadata.title')}</div>
      <dl className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-sm">
        <dt className="text-muted-foreground">{t('profileForm.rvc.metadata.version')}</dt>
        <dd className="text-right font-medium">{profile.rvc_version ?? '—'}</dd>
        <dt className="text-muted-foreground">{t('profileForm.rvc.metadata.sampleRate')}</dt>
        <dd className="text-right font-medium">
          {profile.rvc_sample_rate != null
            ? t('profileForm.rvc.metadata.sampleRateValue', { value: profile.rvc_sample_rate })
            : '—'}
        </dd>
        <dt className="text-muted-foreground">{t('profileForm.rvc.metadata.f0')}</dt>
        <dd className="text-right font-medium">
          {profile.rvc_f0 == null
            ? '—'
            : profile.rvc_f0
              ? t('profileForm.rvc.metadata.f0Yes')
              : t('profileForm.rvc.metadata.f0No')}
        </dd>
      </dl>
    </div>
  );
}

interface RvcModelPickerProps {
  modelFile: File | null;
  indexFile: File | null;
  onModelFileChange: (file: File | null) => void;
  onIndexFileChange: (file: File | null) => void;
  /** Upload fraction (0..1) while the parent form's submit is in flight. */
  uploadFraction?: number | null;
  /** Server-side 400 validator message surfaced from a failed upload. */
  uploadError?: string | null;
  disabled?: boolean;
}

/**
 * Create-mode picker: the parent form owns the selected files and performs the
 * upload after the profile row exists (the upload endpoint needs a profile id).
 */
export function RvcModelPicker({
  modelFile,
  indexFile,
  onModelFileChange,
  onIndexFileChange,
  uploadFraction = null,
  uploadError = null,
  disabled = false,
}: RvcModelPickerProps) {
  const { t } = useTranslation();
  const [clientError, setClientError] = useState<string | null>(null);

  return (
    <div className="space-y-4 pt-4">
      <p className="text-sm text-muted-foreground">{t('profileForm.rvc.createHint')}</p>

      <RvcModelUploadFields
        modelFile={modelFile}
        indexFile={indexFile}
        onModelFile={onModelFileChange}
        onIndexFile={onIndexFileChange}
        onError={setClientError}
        fraction={uploadFraction}
        error={clientError}
        disabled={disabled}
      />

      {uploadError && <RvcErrorBox message={uploadError} />}
    </div>
  );
}

/**
 * Edit-mode manager: shows the status of an existing rvc profile's model and
 * drives live upload / replace / delete against the running backend.
 */
export function RvcModelManager({ profile }: { profile: VoiceProfileResponse }) {
  const { t } = useTranslation();
  const { toast } = useToast();
  const uploadModel = useUploadRvcModel();
  const deleteModel = useDeleteRvcModel();

  const hasModel = profile.rvc_has_model === true;
  const [replacing, setReplacing] = useState(false);
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [indexFile, setIndexFile] = useState<File | null>(null);
  const [fraction, setFraction] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  // Aborts the in-flight upload when the user cancels or the dialog unmounts.
  const abortRef = useRef<AbortController | null>(null);

  // Abort any in-flight upload when this manager unmounts (dialog closed).
  useEffect(() => () => abortRef.current?.abort(), []);

  function resetPicker() {
    setModelFile(null);
    setIndexFile(null);
    setError(null);
    setReplacing(false);
  }

  async function handleUpload() {
    if (!modelFile) {
      setError(t('profileForm.rvc.validation.modelRequired'));
      return;
    }
    setError(null);
    setFraction(0);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await uploadModel.mutateAsync({
        profileId: profile.id,
        modelFile,
        indexFile: indexFile ?? undefined,
        onProgress: (p) => setFraction(p.fraction),
        signal: controller.signal,
      });
      toast({
        title: t('profileForm.rvc.toast.modelUploaded'),
        description: t('profileForm.rvc.toast.modelUploadedDescription'),
      });
      resetPicker();
    } catch (err) {
      // A user-initiated abort is not an error worth surfacing.
      if (controller.signal.aborted) return;
      setError(err instanceof Error ? err.message : t('profileForm.rvc.toast.uploadFailed'));
    } finally {
      abortRef.current = null;
      setFraction(null);
    }
  }

  /** Cancel the picker; also aborts an upload that is still in flight. */
  function handleCancelUpload() {
    abortRef.current?.abort();
    setFraction(null);
    resetPicker();
  }

  async function handleDelete() {
    setError(null);
    try {
      await deleteModel.mutateAsync(profile.id);
      toast({ title: t('profileForm.rvc.toast.modelDeleted') });
      resetPicker();
    } catch (err) {
      const message = err instanceof Error ? err.message : t('profileForm.rvc.toast.deleteFailed');
      setError(message);
      toast({
        title: t('profileForm.rvc.toast.deleteFailed'),
        description: message,
        variant: 'destructive',
      });
    }
  }

  const uploading = fraction != null;

  return (
    <div className="space-y-4 pt-4">
      {hasModel ? (
        <>
          <div className="flex items-center gap-2 text-sm font-medium text-accent">
            <CheckCircle2 className="h-4 w-4" />
            {t('profileForm.rvc.modelReady')}
          </div>
          <RvcMetadata profile={profile} />

          {replacing ? (
            <div className="space-y-4 rounded-lg border border-border p-3">
              <RvcModelUploadFields
                modelFile={modelFile}
                indexFile={indexFile}
                onModelFile={setModelFile}
                onIndexFile={setIndexFile}
                onError={setError}
                fraction={fraction}
                error={error}
                disabled={uploading}
              />
              <div className="flex gap-2">
                <Button type="button" onClick={handleUpload} disabled={uploading || !modelFile}>
                  {uploading ? t('profileForm.rvc.uploadingShort') : t('profileForm.rvc.upload')}
                </Button>
                <Button type="button" variant="outline" onClick={handleCancelUpload}>
                  {t('common.cancel')}
                </Button>
              </div>
            </div>
          ) : (
            <div className="flex gap-2">
              <Button
                type="button"
                variant="outline"
                className="gap-2"
                onClick={() => setReplacing(true)}
              >
                <Replace className="h-4 w-4" />
                {t('profileForm.rvc.replace')}
              </Button>
              <Button
                type="button"
                variant="destructive"
                className="gap-2"
                onClick={() => setConfirmDelete(true)}
                disabled={deleteModel.isPending}
              >
                <Trash2 className="h-4 w-4" />
                {deleteModel.isPending
                  ? t('profileForm.rvc.deleting')
                  : t('profileForm.rvc.delete')}
              </Button>
            </div>
          )}
          {!replacing && error && <RvcErrorBox message={error} />}

          <AlertDialog open={confirmDelete} onOpenChange={setConfirmDelete}>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>{t('profileForm.rvc.deleteDialog.title')}</AlertDialogTitle>
                <AlertDialogDescription>
                  {t('profileForm.rvc.deleteDialog.description')}
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>{t('common.cancel')}</AlertDialogCancel>
                <AlertDialogAction asChild>
                  <Button
                    type="button"
                    onClick={handleDelete}
                    disabled={deleteModel.isPending}
                    className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                  >
                    {deleteModel.isPending
                      ? t('profileForm.rvc.deleting')
                      : t('profileForm.rvc.delete')}
                  </Button>
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </>
      ) : (
        <>
          <div className="text-sm text-muted-foreground">{t('profileForm.rvc.noModel')}</div>
          <RvcModelUploadFields
            modelFile={modelFile}
            indexFile={indexFile}
            onModelFile={setModelFile}
            onIndexFile={setIndexFile}
            onError={setError}
            fraction={fraction}
            error={error}
            disabled={uploading}
          />
          <Button type="button" onClick={handleUpload} disabled={uploading || !modelFile}>
            {uploading ? t('profileForm.rvc.uploadingShort') : t('profileForm.rvc.upload')}
          </Button>
        </>
      )}
    </div>
  );
}

interface RvcChainSettingsProps {
  /** Stored base voice — `"{engine}:{voice_id}"` or `"profile:{profile_id}"`. */
  baseVoice: string;
  params: Required<RvcConvertParams>;
  /** Whether the profile has a feature index — index rate is inert without one. */
  hasIndex: boolean;
  onBaseVoiceChange: (value: string) => void;
  onParamsChange: (params: Required<RvcConvertParams>) => void;
  /**
   * Reports whether the base voice currently resolves to a real, selectable
   * base. The parent form uses this to keep Save from submitting a half-typed
   * `"{engine}:"` / `"profile:"` while the voice or profile list is still
   * loading or errored.
   */
  onValidityChange?: (valid: boolean) => void;
  disabled?: boolean;
}

/** Human-readable label for a base profile's voice type badge. */
function profileTypeKey(voiceType: string): string {
  switch (voiceType) {
    case 'preset':
      return 'profileForm.rvc.chain.profileType.preset';
    case 'designed':
      return 'profileForm.rvc.chain.profileType.designed';
    default:
      return 'profileForm.rvc.chain.profileType.cloned';
  }
}

/**
 * Editor for the TTS→RVC chain: the cloning-free base voice the text is first
 * spoken with, plus the conversion knobs applied afterwards. The base can be a
 * preset engine voice (Kokoro / Qwen CustomVoice) or an existing non-RVC
 * Voicebox profile (a flat select over `useProfiles`) — the latter unlocks
 * languages the presets don't cover (e.g. Russian). Reuses the Voice Changer
 * file-mode `ConversionParams` control so bounds and defaults match.
 */
export function RvcChainSettings({
  baseVoice,
  params,
  hasIndex,
  onBaseVoiceChange,
  onParamsChange,
  onValidityChange,
  disabled = false,
}: RvcChainSettingsProps) {
  const { t } = useTranslation();
  const [source, voiceId] = parseRvcBaseVoice(baseVoice);
  const isProfileBase = source === RVC_PROFILE_SOURCE;

  // Always-fresh mirror of the current source, assigned during render (before
  // any post-commit effect runs). The base-source <Select> onValueChange guard
  // reads this instead of its closure: Radix stores onValueChange in a ref it
  // refreshes in an effect that runs *after* the BubbleSelect echo (see the
  // guard below), so the closure can still hold a stale source at echo time.
  const sourceRef = useRef(source);
  sourceRef.current = source;

  const {
    data: presetData,
    isLoading: voicesLoading,
    isError: voicesError,
  } = useQuery({
    queryKey: ['presetVoices', source],
    queryFn: () => apiClient.listPresetVoices(source),
    enabled: !!source && !isProfileBase,
  });
  const voices = presetData?.voices ?? [];
  const voiceIsValid = !isProfileBase && voices.some((voice: PresetVoice) => voice.voice_id === voiceId);

  // Profile bases: reuse the shared profile-list query. RVC profiles are
  // excluded (one-level recursion guard, mirrored on the backend), so a chain
  // can never be built on another chain.
  const {
    data: profilesData,
    isLoading: profilesLoading,
    isError: profilesError,
  } = useProfiles();
  const baseProfiles = (profilesData ?? []).filter(
    (profile: VoiceProfileResponse) => profile.voice_type !== 'rvc',
  );
  const profileIsValid =
    isProfileBase && baseProfiles.some((profile: VoiceProfileResponse) => profile.id === voiceId);

  // After switching engines the stored voice_id no longer exists — snap to the
  // first available voice so the persisted base voice is always valid.
  useEffect(() => {
    if (isProfileBase || voices.length === 0 || voiceIsValid) return;
    onBaseVoiceChange(`${source}:${voices[0].voice_id}`);
  }, [voices, voiceIsValid, source, isProfileBase, onBaseVoiceChange]);

  // Same convergence for the profile source: an empty/stale `profile:{id}` snaps
  // to the first available non-RVC profile.
  useEffect(() => {
    if (!isProfileBase || baseProfiles.length === 0 || profileIsValid) return;
    onBaseVoiceChange(`${RVC_PROFILE_SOURCE}:${baseProfiles[0].id}`);
  }, [isProfileBase, baseProfiles, profileIsValid, onBaseVoiceChange]);

  // The base is "resolved" once the relevant list has loaded without error and
  // the current id exists in it (the effects above guarantee convergence). A
  // profile source with no eligible profiles never resolves, which correctly
  // blocks Save until the user picks a preset or creates a base profile.
  const baseVoiceResolved = isProfileBase
    ? !profilesLoading && !profilesError && profileIsValid
    : !voicesLoading && !voicesError && voices.length > 0 && voiceIsValid;
  useEffect(() => {
    onValidityChange?.(baseVoiceResolved);
  }, [baseVoiceResolved, onValidityChange]);

  return (
    <div className="space-y-4 rounded-lg border border-border p-4">
      <div className="space-y-1">
        <div className="text-sm font-medium">{t('profileForm.rvc.chain.title')}</div>
        <p className="text-xs text-muted-foreground">{t('profileForm.rvc.chain.hint')}</p>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1.5">
          <Label>{t('profileForm.rvc.chain.baseSource')}</Label>
          <Select
            value={source}
            // Only reset the voice id when the source *genuinely* changes.
            // Radix's hidden BubbleSelect re-dispatches a native `change` event
            // whenever the controlled value updates programmatically, which
            // re-fires onValueChange with the current source — e.g. when editing
            // a profile-based RVC voice loads `profile:{id}` and flips the source
            // from the default `kokoro` to `profile`. Without this guard that
            // echo runs `${value}:`, wiping the just-loaded id and snapping the
            // base voice back to the first available profile. Compare against
            // `sourceRef` (fresh as of the last render), not the closure, which
            // Radix may still hold from a stale render at echo time.
            onValueChange={(value) => {
              if (value !== sourceRef.current) onBaseVoiceChange(`${value}:`);
            }}
            disabled={disabled}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {RVC_BASE_ENGINES.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
              <SelectItem value={RVC_PROFILE_SOURCE}>
                {t('profileForm.rvc.chain.profileSource')}
              </SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1.5">
          <Label>{t('profileForm.rvc.chain.baseVoice')}</Label>
          {isProfileBase ? (
            <Select
              value={profileIsValid ? voiceId : ''}
              onValueChange={(value) => {
                // Radix's hidden BubbleSelect re-dispatches a change event when
                // the controlled value transitions programmatically — including
                // to '' while the list is still loading. A real item click can
                // never produce '' (Radix forbids empty SelectItem values), so
                // an empty echo must not wipe the id (`profile:`), which the
                // snap effect would then "converge" to the first profile.
                if (!value) return;
                onBaseVoiceChange(`${RVC_PROFILE_SOURCE}:${value}`);
              }}
              disabled={disabled || baseProfiles.length === 0}
            >
              <SelectTrigger>
                <SelectValue placeholder={t('profileForm.rvc.chain.baseProfilePlaceholder')} />
              </SelectTrigger>
              <SelectContent>
                {baseProfiles.map((profile: VoiceProfileResponse) => (
                  <SelectItem key={profile.id} value={profile.id}>
                    <span className="flex items-center gap-1.5">
                      <span>{`${profile.name} — ${profile.language}`}</span>
                      <Badge variant="outline" className="text-[10px] h-4 px-1">
                        {t(profileTypeKey(profile.voice_type))}
                      </Badge>
                    </span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Select
              value={voiceIsValid ? voiceId : ''}
              onValueChange={(value) => {
                // Same empty-echo guard as the profile select above.
                if (!value) return;
                onBaseVoiceChange(`${source}:${value}`);
              }}
              disabled={disabled || voices.length === 0}
            >
              <SelectTrigger>
                <SelectValue placeholder={t('profileForm.rvc.chain.baseVoicePlaceholder')} />
              </SelectTrigger>
              <SelectContent>
                {voices.map((voice: PresetVoice) => (
                  <SelectItem key={voice.voice_id} value={voice.voice_id}>
                    {`${voice.name} — ${voice.language}`}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </div>
      </div>

      {/* Coverage gap: preset engines cover a fixed set of languages. Point users
          who need an uncovered language (e.g. Russian) at a cloned profile base. */}
      {isProfileBase ? (
        baseProfiles.length === 0 &&
        !profilesLoading && (
          <p className="text-xs text-muted-foreground">
            {t('profileForm.rvc.chain.noProfiles')}
          </p>
        )
      ) : (
        <p className="text-xs text-muted-foreground">{t('profileForm.rvc.chain.coverageHint')}</p>
      )}

      <ConversionParams
        f0UpKey={params.f0_up_key}
        onF0UpKey={(value) => onParamsChange({ ...params, f0_up_key: value })}
        f0Method={params.f0_method}
        onF0Method={(value) => onParamsChange({ ...params, f0_method: value })}
        indexRate={params.index_rate}
        onIndexRate={(value) => onParamsChange({ ...params, index_rate: value })}
        rmsMixRate={params.rms_mix_rate}
        onRmsMixRate={(value) => onParamsChange({ ...params, rms_mix_rate: value })}
        protect={params.protect}
        onProtect={(value) => onParamsChange({ ...params, protect: value })}
        hasIndex={hasIndex}
        disabled={disabled}
      />
    </div>
  );
}
