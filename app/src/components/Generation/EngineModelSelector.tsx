import { useEffect } from 'react';
import type { UseFormReturn } from 'react-hook-form';
import { FormControl } from '@/components/ui/form';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import type { VoiceProfileResponse } from '@/lib/api/types';
import { getLanguageOptionsForEngine } from '@/lib/constants/languages';
import type { GenerationFormValues } from '@/lib/hooks/useGenerationForm';

/**
 * Engine/model options and their display metadata.
 * Adding a new engine means adding one entry here.
 */
const ENGINE_OPTIONS = [
  { value: 'qwen:1.7B', label: 'Qwen3-TTS 1.7B', engine: 'qwen' },
  { value: 'qwen:0.6B', label: 'Qwen3-TTS 0.6B', engine: 'qwen' },
  { value: 'qwen_custom_voice:1.7B', label: 'Qwen CustomVoice 1.7B', engine: 'qwen_custom_voice' },
  { value: 'qwen_custom_voice:0.6B', label: 'Qwen CustomVoice 0.6B', engine: 'qwen_custom_voice' },
  { value: 'luxtts', label: 'LuxTTS', engine: 'luxtts' },
  { value: 'chatterbox', label: 'Chatterbox Multilingual', engine: 'chatterbox' },
  { value: 'chatterbox_turbo', label: 'Chatterbox Turbo (English)', engine: 'chatterbox_turbo' },
  { value: 'tada:1B', label: 'TADA 1B', engine: 'tada' },
  { value: 'tada:3B', label: 'TADA 3B Multilingual', engine: 'tada' },
  { value: 'kokoro', label: 'Kokoro 82M', engine: 'kokoro' },
  { value: 'voxtral', label: 'Voxtral 4B TTS', engine: 'voxtral' },
  { value: 'rvc', label: 'RVC', engine: 'rvc' },
] as const;

const ENGINE_DESCRIPTIONS: Record<string, string> = {
  qwen: 'Multi-language, two sizes',
  qwen_custom_voice: '9 preset voices, instruct control',
  luxtts: 'Fast, English-focused',
  chatterbox: '23 languages, incl. Hebrew',
  chatterbox_turbo: 'English, [laugh] [cough] tags',
  tada: 'HumeAI, 700s+ coherent audio',
  kokoro: '82M params, CPU realtime, 8 langs',
  voxtral: 'MLX 4-bit, 20 preset voices',
  rvc: 'Voice conversion chain',
};

/** Engines that only support English and should force language to 'en' on select. */
const ENGLISH_ONLY_ENGINES = new Set(['luxtts', 'chatterbox_turbo']);

/** Engines that support cloned (reference audio) profiles. */
const CLONING_ENGINES = new Set(['qwen', 'luxtts', 'chatterbox', 'chatterbox_turbo', 'tada']);

type ModelSize = NonNullable<GenerationFormValues['modelSize']>;

function getAvailableOptions() {
  // Engine selection is the source of truth. Profile compatibility is handled
  // by the profile list and picker, so selecting a new engine can reveal its
  // voices instead of being blocked by the currently selected profile.
  return ENGINE_OPTIONS;
}

export function resolveModelSizeForEngine(
  engine: string,
  modelSize?: string,
  language?: string,
): ModelSize | undefined {
  if (engine === 'qwen' || engine === 'qwen_custom_voice') {
    return modelSize === '0.6B' ? '0.6B' : '1.7B';
  }
  if (engine === 'tada') {
    if (language && language !== 'en') return '3B';
    if (modelSize === '1B' || modelSize === '3B') return modelSize;
    return '1B';
  }
  return undefined;
}

function getSelectValue(engine: string, modelSize?: string, language?: string): string {
  const resolvedModelSize = resolveModelSizeForEngine(engine, modelSize, language);
  if (engine === 'qwen') return `qwen:${resolvedModelSize || '1.7B'}`;
  if (engine === 'qwen_custom_voice') return `qwen_custom_voice:${resolvedModelSize || '1.7B'}`;
  if (engine === 'tada') return `tada:${resolvedModelSize || '1B'}`;
  return engine;
}

export function applyEngineSelection(form: UseFormReturn<GenerationFormValues>, value: string) {
  if (value.startsWith('qwen_custom_voice:')) {
    const [, modelSize] = value.split(':');
    form.setValue('engine', 'qwen_custom_voice');
    form.setValue('modelSize', modelSize as '1.7B' | '0.6B');
    const currentLang = form.getValues('language');
    const available = getLanguageOptionsForEngine('qwen_custom_voice');
    if (!available.some((l) => l.value === currentLang)) {
      form.setValue('language', available[0]?.value ?? 'en');
    }
  } else if (value.startsWith('qwen:')) {
    const [, modelSize] = value.split(':');
    form.setValue('engine', 'qwen');
    form.setValue('modelSize', modelSize as '1.7B' | '0.6B');
    // Validate language is supported by Qwen
    const currentLang = form.getValues('language');
    const available = getLanguageOptionsForEngine('qwen');
    if (!available.some((l) => l.value === currentLang)) {
      form.setValue('language', available[0]?.value ?? 'en');
    }
  } else if (value.startsWith('tada:')) {
    const [, modelSize] = value.split(':');
    form.setValue('engine', 'tada');
    form.setValue('modelSize', modelSize as '1B' | '3B');
    // TADA 1B is English-only; 3B is multilingual
    if (modelSize === '1B') {
      form.setValue('language', 'en');
    } else {
      const currentLang = form.getValues('language');
      const available = getLanguageOptionsForEngine('tada');
      if (!available.some((l) => l.value === currentLang)) {
        form.setValue('language', available[0]?.value ?? 'en');
      }
    }
  } else if (value === 'rvc') {
    form.setValue('engine', 'rvc');
  } else {
    form.setValue('engine', value as GenerationFormValues['engine']);
    form.setValue('modelSize', undefined);
    if (ENGLISH_ONLY_ENGINES.has(value)) {
      form.setValue('language', 'en');
    } else {
      // If current language isn't supported by the new engine, reset to first available
      const currentLang = form.getValues('language');
      const available = getLanguageOptionsForEngine(value);
      if (!available.some((l) => l.value === currentLang)) {
        form.setValue('language', available[0]?.value ?? 'en');
      }
    }
  }
}

interface EngineModelSelectorProps {
  form: UseFormReturn<GenerationFormValues>;
  compact?: boolean;
}

export function EngineModelSelector({ form, compact }: EngineModelSelectorProps) {
  const engine = form.watch('engine') || 'qwen';
  const modelSize = form.watch('modelSize');
  const language = form.watch('language');
  const resolvedModelSize = resolveModelSizeForEngine(engine, modelSize, language);
  const selectValue = getSelectValue(engine, modelSize, language);
  const availableOptions = getAvailableOptions();

  const currentEngineAvailable = availableOptions.some((opt) => opt.value === selectValue);

  useEffect(() => {
    if (engine !== 'rvc' && resolvedModelSize !== modelSize) {
      form.setValue('modelSize', resolvedModelSize);
    }
  }, [engine, form, modelSize, resolvedModelSize]);

  useEffect(() => {
    if (!currentEngineAvailable && availableOptions.length > 0) {
      const sameEngineOption = availableOptions.find((opt) => opt.engine === engine);
      applyEngineSelection(form, sameEngineOption?.value ?? availableOptions[0].value);
    }
  }, [availableOptions, currentEngineAvailable, engine, form]);

  const itemClass = compact ? 'text-xs text-muted-foreground' : undefined;
  const triggerClass = compact
    ? 'h-8 text-xs bg-card border-border rounded-full hover:bg-background/50 transition-all'
    : undefined;

  return (
    <Select value={selectValue} onValueChange={(v) => applyEngineSelection(form, v)}>
      <FormControl>
        <SelectTrigger className={triggerClass}>
          <SelectValue />
        </SelectTrigger>
      </FormControl>
      <SelectContent side={compact ? 'top' : undefined}>
        {availableOptions.map((opt) => (
          <SelectItem key={opt.value} value={opt.value} className={itemClass}>
            {opt.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

/** Returns a human-readable description for the currently selected engine. */
export function getEngineDescription(engine: string): string {
  return ENGINE_DESCRIPTIONS[engine] ?? '';
}

/**
 * Check if a profile is compatible with the currently selected engine.
 * Useful for UI hints.
 */
export function isProfileCompatibleWithEngine(
  profile: VoiceProfileResponse,
  engine: string,
): boolean {
  const voiceType = profile.voice_type || 'cloned';
  if (voiceType === 'preset') return profile.preset_engine === engine;
  if (voiceType === 'cloned') return CLONING_ENGINES.has(engine);
  if (voiceType === 'rvc') return engine === 'rvc';
  return true; // designed — future
}
