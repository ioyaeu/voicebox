import { zodResolver } from '@hookform/resolvers/zod';
import { useState } from 'react';
import { useForm } from 'react-hook-form';
import * as z from 'zod';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import type { EffectConfig, ModelStatus } from '@/lib/api/types';
import { LANGUAGE_CODES, type LanguageCode } from '@/lib/constants/languages';
import { useGeneration } from '@/lib/hooks/useGeneration';
import { useModelDownloadToast } from '@/lib/hooks/useModelDownloadToast';
import { useGenerationSettings } from '@/lib/hooks/useSettings';
import { useGenerationStore } from '@/stores/generationStore';
import { useUIStore } from '@/stores/uiStore';

const generationSchema = z.object({
  text: z.string().min(1, '').max(50000),
  language: z.enum(LANGUAGE_CODES as [LanguageCode, ...LanguageCode[]]),
  seed: z.number().int().optional(),
  modelSize: z.enum(['1.7B', '0.6B', '1B', '3B']).optional(),
  instruct: z.string().max(500).optional(),
  engine: z
    .enum([
      'qwen',
      'qwen_custom_voice',
      'luxtts',
      'chatterbox',
      'chatterbox_turbo',
      'tada',
      'kokoro',
      'voxtral',
      'rvc',
    ])
    .optional(),
  personality: z.boolean().optional(),
});

export type GenerationFormValues = z.infer<typeof generationSchema>;

/**
 * Maps an RVC chain base voice's engine to the downloadable base TTS model, so
 * an RVC generation can show the same base-model download dialog a plain TTS
 * generation would (the chain speaks the text with this base voice first). The
 * backend resolves the authoritative model/size at run time; this drives the
 * client-side download toast for the common base engines.
 */
function resolveRvcBaseModel(
  baseEngine: string,
  modelSize: GenerationFormValues['modelSize'],
): { modelName: string; displayName: string } | null {
  if (baseEngine === 'kokoro') {
    return { modelName: 'kokoro', displayName: 'Kokoro 82M' };
  }
  if (baseEngine === 'voxtral') {
    return { modelName: 'voxtral-4b-tts-4bit', displayName: 'Voxtral 4B TTS' };
  }
  if (baseEngine === 'qwen_custom_voice') {
    const size = modelSize === '0.6B' ? '0.6B' : '1.7B';
    return {
      modelName: `qwen-custom-voice-${size}`,
      displayName: `Qwen CustomVoice ${size}`,
    };
  }
  return null;
}

function findModelForEngine(
  models: ModelStatus[],
  engine: string,
  modelSize: GenerationFormValues['modelSize'],
): ModelStatus | undefined {
  return models.find((model) => {
    if (model.engine !== engine) return false;
    if (engine === 'qwen' || engine === 'qwen_custom_voice' || engine === 'tada') {
      return model.model_size === modelSize;
    }
    return true;
  });
}

function resolveModelSizeForRequest(
  engine: string,
  modelSize: GenerationFormValues['modelSize'],
  language: LanguageCode,
): GenerationFormValues['modelSize'] {
  if (engine === 'qwen' || engine === 'qwen_custom_voice') {
    return modelSize === '0.6B' ? '0.6B' : '1.7B';
  }
  if (engine === 'tada') {
    if (language !== 'en') return '3B';
    return modelSize === '3B' ? '3B' : '1B';
  }
  return undefined;
}

interface UseGenerationFormOptions {
  onSuccess?: (generationId: string) => void;
  defaultValues?: Partial<GenerationFormValues>;
  getEffectsChain?: () => EffectConfig[] | undefined;
  /** When the selected profile is an rvc voice, send engine=null so the backend
   *  resolves the TTS→RVC chain — the profile owns the base engine, and "rvc"
   *  is not a valid TTS engine on the request. */
  isRvcProfile?: boolean;
}

export function useGenerationForm(options: UseGenerationFormOptions = {}) {
  const { toast } = useToast();
  const generation = useGeneration();
  const addPendingGeneration = useGenerationStore((state) => state.addPendingGeneration);
  const { settings: genSettings } = useGenerationSettings();
  const maxChunkChars = genSettings?.max_chunk_chars ?? 800;
  const crossfadeMs = genSettings?.crossfade_ms ?? 50;
  const normalizeAudio = genSettings?.normalize_audio ?? true;
  const selectedEngine = useUIStore((state) => state.selectedEngine);
  const [downloadingModelName, setDownloadingModelName] = useState<string | null>(null);
  const [downloadingDisplayName, setDownloadingDisplayName] = useState<string | null>(null);

  useModelDownloadToast({
    modelName: downloadingModelName || '',
    displayName: downloadingDisplayName || '',
    enabled: !!downloadingModelName,
  });

  const form = useForm<GenerationFormValues>({
    resolver: zodResolver(generationSchema),
    defaultValues: {
      text: '',
      language: 'en',
      seed: undefined,
      modelSize: '1.7B',
      instruct: '',
      engine: (selectedEngine as GenerationFormValues['engine']) || 'qwen',
      personality: false,
      ...options.defaultValues,
    },
  });

  async function handleSubmit(
    data: GenerationFormValues,
    selectedProfileId: string | null,
  ): Promise<void> {
    if (!selectedProfileId) {
      toast({
        title: 'No profile selected',
        description: 'Please select a voice profile from the cards above.',
        variant: 'destructive',
      });
      return;
    }

    try {
      // RVC profiles resolve their base engine server-side; the request sends
      // engine=null. The base voice's TTS model still needs downloading, so we
      // surface the normal download dialog for it below rather than skipping it.
      const isRvc = options.isRvcProfile === true;
      const engine = data.engine || 'qwen';
      const requestModelSize = resolveModelSizeForRequest(engine, data.modelSize, data.language);

      if (!isRvc) {
        // Check if model needs downloading
        try {
          const modelStatus = await apiClient.getModelStatus();
          const model = findModelForEngine(modelStatus.models, engine, requestModelSize);

          if (model && !model.downloaded) {
            setDownloadingModelName(model.model_name);
            setDownloadingDisplayName(model.display_name);
          }
        } catch (error) {
          console.error('Failed to check model status:', error);
        }
      } else {
        // RVC: resolve the chain's base TTS model from the profile's base voice
        // and show the same download dialog if it isn't present yet.
        try {
          const profile = await apiClient.getProfile(selectedProfileId);
          const baseEngine = profile.rvc_base_voice?.split(':')[0] || 'kokoro';
          const base = resolveRvcBaseModel(baseEngine, data.modelSize);
          if (base) {
            const modelStatus = await apiClient.getModelStatus();
            const model = modelStatus.models.find((m) => m.model_name === base.modelName);
            if (model && !model.downloaded) {
              setDownloadingModelName(base.modelName);
              setDownloadingDisplayName(base.displayName);
            }
          }
        } catch (error) {
          console.error('Failed to check RVC base model status:', error);
        }
      }

      const hasModelSizes =
        engine === 'qwen' || engine === 'qwen_custom_voice' || engine === 'tada';
      // Only Qwen CustomVoice actually honors the instruct kwarg at model level.
      // Base Qwen3-TTS accepts the kwarg but ignores it.
      const supportsInstruct = engine === 'qwen_custom_voice';
      const effectsChain = options.getEffectsChain?.();
      // This now returns immediately with status="generating"
      const result = await generation.mutateAsync({
        profile_id: selectedProfileId,
        text: data.text,
        language: data.language,
        seed: data.seed,
        model_size: isRvc ? undefined : hasModelSizes ? requestModelSize : undefined,
        // null defers the engine choice to the profile (rvc TTS→RVC chain).
        engine: isRvc ? null : engine,
        instruct: isRvc ? undefined : supportsInstruct ? data.instruct || undefined : undefined,
        personality: data.personality || undefined,
        max_chunk_chars: maxChunkChars,
        crossfade_ms: crossfadeMs,
        normalize: normalizeAudio,
        effects_chain: effectsChain?.length ? effectsChain : undefined,
      });

      // Track this generation for SSE status updates
      addPendingGeneration(result.id);

      // Reset form immediately — user can start typing again
      form.reset({
        text: '',
        language: data.language,
        seed: undefined,
        modelSize: requestModelSize,
        instruct: '',
        engine: data.engine,
        personality: data.personality,
      });
      options.onSuccess?.(result.id);
    } catch (error) {
      toast({
        title: 'Generation failed',
        description: error instanceof Error ? error.message : 'Failed to generate audio',
        variant: 'destructive',
      });
    } finally {
      setDownloadingModelName(null);
      setDownloadingDisplayName(null);
    }
  }

  return {
    form,
    handleSubmit,
    isPending: generation.isPending,
  };
}
