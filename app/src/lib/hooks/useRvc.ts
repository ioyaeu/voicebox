import { useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';
import type { RvcConvertParams, UploadProgress } from '@/lib/api/types';
import { useGenerationStore } from '@/stores/generationStore';

export function useUploadRvcModel() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      profileId,
      modelFile,
      indexFile,
      onProgress,
      signal,
    }: {
      profileId: string;
      modelFile: File;
      indexFile?: File;
      onProgress?: (progress: UploadProgress) => void;
      /** Aborts the in-flight upload (e.g. the profile dialog closing). */
      signal?: AbortSignal;
    }) => apiClient.uploadRvcModel(profileId, modelFile, indexFile, onProgress, signal),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['profiles'] });
      queryClient.invalidateQueries({ queryKey: ['profiles', variables.profileId] });
    },
  });
}

export function useDeleteRvcModel() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (profileId: string) => apiClient.deleteRvcModel(profileId),
    onSuccess: (_, profileId) => {
      queryClient.invalidateQueries({ queryKey: ['profiles'] });
      queryClient.invalidateQueries({ queryKey: ['profiles', profileId] });
    },
  });
}

export function useStartConversion() {
  const queryClient = useQueryClient();
  const addPendingGeneration = useGenerationStore((s) => s.addPendingGeneration);

  return useMutation({
    mutationFn: ({
      file,
      profileId,
      params,
    }: {
      file: File;
      profileId: string;
      params?: RvcConvertParams;
    }) => apiClient.startConversion(file, profileId, params),
    onSuccess: (result) => {
      // Conversion tasks share the TTS task pipeline: registering the task id
      // lets the globally-mounted useGenerationProgress hook stream SSE status,
      // refetch history, and autoplay the result on completion. Callers can
      // watch generationStore.pendingGenerationIds for this task_id and read
      // the finished row via useGenerationDetail(task_id).
      addPendingGeneration(result.task_id);
      queryClient.invalidateQueries({ queryKey: ['history'] });
    },
  });
}
