import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useCallback, useRef } from 'react';
import { apiClient } from '@/lib/api/client';
import type {
  MCPClientBinding,
  MCPClientBindingListResponse,
  MCPClientBindingUpsert,
} from '@/lib/api/types';

const MCP_BINDINGS_KEY = ['settings', 'mcp', 'bindings'] as const;

function toUpsert(binding: MCPClientBinding): MCPClientBindingUpsert {
  return {
    client_id: binding.client_id,
    label: binding.label,
    profile_id: binding.profile_id,
    default_engine: binding.default_engine,
    default_personality: binding.default_personality,
    default_plain_text: binding.default_plain_text,
    default_max_chars: binding.default_max_chars,
  };
}

/** Manage per-MCP-client voice bindings (Claude Code → Morgan, etc.). */
export function useMCPBindings() {
  const queryClient = useQueryClient();
  const pendingUpdates = useRef(new Map<string, MCPClientBindingUpsert>());
  const updateQueues = useRef(new Map<string, Promise<void>>());

  const query = useQuery({
    queryKey: MCP_BINDINGS_KEY,
    queryFn: () => apiClient.listMCPBindings(),
    // Keep fresh while the Settings page is open — the ``last_seen_at``
    // timestamp is useful for confirming an install works, and we want it
    // to tick forward when a client connects.
    refetchInterval: 10_000,
  });

  const upsertMutation = useMutation({
    mutationFn: (data: MCPClientBindingUpsert) => apiClient.upsertMCPBinding(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: MCP_BINDINGS_KEY });
    },
  });

  const updateAsync = useCallback(
    (binding: MCPClientBinding, patch: Partial<MCPClientBindingUpsert>) => {
      const clientId = binding.client_id;
      const cached = queryClient
        .getQueryData<MCPClientBindingListResponse>(MCP_BINDINGS_KEY)
        ?.items.find((item) => item.client_id === clientId);
      const base = pendingUpdates.current.get(clientId) ?? toUpsert(cached ?? binding);
      const next = { ...base, ...patch };
      pendingUpdates.current.set(clientId, next);

      if (!updateQueues.current.has(clientId)) {
        const queue = (async () => {
          try {
            while (true) {
              const payload = pendingUpdates.current.get(clientId);
              if (!payload) break;

              const saved = await apiClient.upsertMCPBinding(payload);
              queryClient.setQueryData<MCPClientBindingListResponse>(MCP_BINDINGS_KEY, (data) =>
                data
                  ? {
                      items: data.items.map((item) => (item.client_id === clientId ? saved : item)),
                    }
                  : data,
              );

              if (pendingUpdates.current.get(clientId) === payload) {
                pendingUpdates.current.delete(clientId);
                break;
              }
            }
          } finally {
            updateQueues.current.delete(clientId);
          }

          await queryClient.invalidateQueries({ queryKey: MCP_BINDINGS_KEY });
        })();
        updateQueues.current.set(clientId, queue);
      }

      return updateQueues.current.get(clientId);
    },
    [queryClient],
  );

  const deleteMutation = useMutation({
    mutationFn: (clientId: string) => apiClient.deleteMCPBinding(clientId),
    onMutate: async (clientId) => {
      await queryClient.cancelQueries({ queryKey: MCP_BINDINGS_KEY });
      const prev = queryClient.getQueryData<MCPClientBindingListResponse>(MCP_BINDINGS_KEY);
      if (prev) {
        queryClient.setQueryData<MCPClientBindingListResponse>(MCP_BINDINGS_KEY, {
          items: prev.items.filter((b) => b.client_id !== clientId),
        });
      }
      return { prev };
    },
    onError: (_err, _id, ctx) => {
      if (ctx?.prev) queryClient.setQueryData(MCP_BINDINGS_KEY, ctx.prev);
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: MCP_BINDINGS_KEY });
    },
  });

  return {
    bindings: query.data?.items ?? [],
    isLoading: query.isLoading,
    upsert: upsertMutation.mutate,
    upsertAsync: upsertMutation.mutateAsync,
    updateAsync,
    remove: deleteMutation.mutate,
  };
}
