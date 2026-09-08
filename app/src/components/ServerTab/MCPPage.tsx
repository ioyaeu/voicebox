import { Check, Copy, Plug, Radio, RotateCcw, Terminal, Trash2 } from 'lucide-react';
import { useState } from 'react';
import { Trans, useTranslation } from 'react-i18next';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import type { MCPClientBinding, MCPClientBindingUpsert } from '@/lib/api/types';
import { useMCPBindings } from '@/lib/hooks/useMCPBindings';
import { useProfiles } from '@/lib/hooks/useProfiles';
import { useCaptureSettings } from '@/lib/hooks/useSettings';
import { formatDate } from '@/lib/utils/format';
import { useServerStore } from '@/stores/serverStore';
import { SettingRow, SettingSection } from './SettingRow';

function getStdioShimCommand(): string {
  if (typeof navigator === 'undefined') {
    return '/Applications/Voicebox.app/Contents/MacOS/voicebox-mcp';
  }

  const platform = `${navigator.platform} ${navigator.userAgent}`.toLowerCase();
  if (platform.includes('win')) {
    return 'C:\\Program Files\\Voicebox\\voicebox-mcp.exe';
  }
  if (platform.includes('linux')) {
    return '/opt/voicebox/voicebox-mcp';
  }
  return '/Applications/Voicebox.app/Contents/MacOS/voicebox-mcp';
}

const SPEECH_SESSION_TOOLS = [
  'voicebox.speech_start',
  'voicebox.speech_append',
  'voicebox.speech_finish',
  'voicebox.speech_status',
  'voicebox.speech_cancel',
] as const;

/**
 * Settings -> MCP — configure per-agent voice binding and show copy-paste
 * install snippets for major MCP clients and the local VS Code narration
 * bridge prototype.
 */
export function MCPPage() {
  const { t } = useTranslation();
  const serverUrl = useServerStore((s) => s.serverUrl);
  const { bindings, upsertAsync, updateAsync, remove } = useMCPBindings();
  const { data: profiles } = useProfiles();
  const { settings: captureSettings, update: updateCapture } = useCaptureSettings();

  const defaultProfileId = captureSettings?.default_playback_voice_id ?? '';
  const mcpUrl = `${serverUrl}/mcp`;
  const stdioShimCommand = getStdioShimCommand();
  const restSessionSnippet = [
    `POST ${serverUrl}/speak/sessions`,
    `POST ${serverUrl}/speak/sessions/{session_id}/append`,
    `POST ${serverUrl}/speak/sessions/{session_id}/finish`,
    `GET  ${serverUrl}/speak/sessions/{session_id}`,
    `POST ${serverUrl}/speak/sessions/{session_id}/cancel`,
  ].join('\n');

  const [newClientId, setNewClientId] = useState('');
  const [newLabel, setNewLabel] = useState('');
  const [newProfileId, setNewProfileId] = useState('');
  const [adding, setAdding] = useState(false);

  const saveBinding = (b: MCPClientBinding, patch: Partial<MCPClientBindingUpsert>) =>
    updateAsync(b, patch);

  const commitMaxChars = (b: MCPClientBinding, raw: string) => {
    const trimmed = raw.trim();
    const next = trimmed === '' ? null : Math.min(10000, Math.max(50, Number(trimmed) || 50));
    if (next !== b.default_max_chars) saveBinding(b, { default_max_chars: next });
  };

  const handleAdd = async () => {
    if (!newClientId.trim()) return;
    setAdding(true);
    try {
      await upsertAsync({
        client_id: newClientId.trim(),
        label: newLabel.trim() || null,
        profile_id: newProfileId || null,
      });
      setNewClientId('');
      setNewLabel('');
      setNewProfileId('');
    } finally {
      setAdding(false);
    }
  };

  return (
    <div className="flex gap-8 items-start max-w-5xl">
      <div className="flex-1 min-w-0 max-w-2xl space-y-8">
        <SettingSection
          title={t('settings.mcp.install.title')}
          description={t('settings.mcp.install.description')}
        >
          <SnippetRow
            title={t('settings.mcp.install.http.title')}
            description={t('settings.mcp.install.http.description')}
            snippet={JSON.stringify(
              {
                mcpServers: {
                  voicebox: {
                    url: mcpUrl,
                    headers: { 'X-Voicebox-Client-Id': 'claude-code' },
                  },
                },
              },
              null,
              2,
            )}
          />
          <SnippetRow
            title={t('settings.mcp.install.claudeCode.title')}
            description={t('settings.mcp.install.claudeCode.description')}
            snippet={`claude mcp add voicebox --transport http --url ${mcpUrl} --header "X-Voicebox-Client-Id: claude-code"`}
          />
          <SnippetRow
            title={t('settings.mcp.install.stdio.title')}
            description={t('settings.mcp.install.stdio.description')}
            snippet={JSON.stringify(
              {
                mcpServers: {
                  voicebox: {
                    command: stdioShimCommand,
                    env: { VOICEBOX_CLIENT_ID: 'claude-code' },
                  },
                },
              },
              null,
              2,
            )}
          />
        </SettingSection>

        <SettingSection
          title={t('settings.mcp.sessions.title')}
          description={t('settings.mcp.sessions.description')}
        >
          <SettingRow
            title={t('settings.mcp.sessions.tools.title')}
            description={t('settings.mcp.sessions.tools.description')}
            action={<Badge variant="secondary">{t('settings.mcp.sessions.tools.badge')}</Badge>}
          >
            <ToolList tools={SPEECH_SESSION_TOOLS} />
          </SettingRow>
          <SnippetRow
            title={t('settings.mcp.sessions.flow.title')}
            description={t('settings.mcp.sessions.flow.description')}
            snippet={[
              'voicebox.speech_start(keep_audio=false)',
              'voicebox.speech_append(session_id, sequence=0, text="First sentence.")',
              'voicebox.speech_append(session_id, sequence=1, text="Next sentence.")',
              'voicebox.speech_finish(session_id)',
              'voicebox.speech_status(session_id)',
            ].join('\n')}
          />
          <SnippetRow
            title={t('settings.mcp.sessions.rest.title')}
            description={t('settings.mcp.sessions.rest.description')}
            snippet={restSessionSnippet}
          />
        </SettingSection>

        <SettingSection
          title={t('settings.mcp.agentBridge.title')}
          description={t('settings.mcp.agentBridge.description')}
        >
          <SettingRow
            title={t('settings.mcp.agentBridge.bindings.title')}
            description={t('settings.mcp.agentBridge.bindings.description')}
            action={<Badge variant="outline">{t('settings.mcp.agentBridge.bindings.badge')}</Badge>}
          >
            <ToolList tools={['claude-code', 'codex-vscode']} />
          </SettingRow>
          <SnippetRow
            title={t('settings.mcp.agentBridge.preview.title')}
            description={t('settings.mcp.agentBridge.preview.description')}
            snippet="node scripts/install-agent-bridge.mjs --claude --codex-experimental"
          />
          <SnippetRow
            title={t('settings.mcp.agentBridge.install.title')}
            description={t('settings.mcp.agentBridge.install.description')}
            snippet="node scripts/install-agent-bridge.mjs --claude --codex-experimental --apply"
          />
          <SnippetRow
            title={t('settings.mcp.agentBridge.restore.title')}
            description={t('settings.mcp.agentBridge.restore.description')}
            snippet="node scripts/install-agent-bridge.mjs --restore --apply"
          />
          <SettingRow
            title={t('settings.mcp.agentBridge.runtime.title')}
            description={t('settings.mcp.agentBridge.runtime.description')}
            action={<RotateCcw className="h-4 w-4 text-muted-foreground" />}
          >
            <code className="block text-[11px] p-3 rounded-md bg-muted/50 break-all">
              ~/.local/share/voicebox/agent-bridge/config.json
            </code>
          </SettingRow>
        </SettingSection>

        <SettingSection
          title={t('settings.mcp.defaultVoice.title')}
          description={t('settings.mcp.defaultVoice.description')}
        >
          <SettingRow
            title={t('settings.mcp.defaultVoice.label')}
            description={t('settings.mcp.defaultVoice.labelHint')}
            action={
              <Select
                value={defaultProfileId || '__default__'}
                onValueChange={(v) =>
                  updateCapture({
                    default_playback_voice_id: v === '__default__' ? null : v,
                  })
                }
              >
                <SelectTrigger className="w-[220px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__default__">{t('settings.mcp.defaultVoice.none')}</SelectItem>
                  {(profiles ?? []).map((p) => (
                    <SelectItem key={p.id} value={p.id}>
                      {p.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            }
          />
        </SettingSection>

        <SettingSection
          title={t('settings.mcp.bindings.title')}
          description={t('settings.mcp.bindings.description')}
        >
          {bindings.length === 0 ? (
            <p className="text-sm text-muted-foreground py-4 italic">
              <Trans i18nKey="settings.mcp.bindings.empty" components={{ code: <code /> }} />
            </p>
          ) : (
            <div className="divide-y divide-border/60">
              {bindings.map((b) => (
                <div
                  key={b.client_id}
                  className="py-3 grid grid-cols-1 gap-3 items-center sm:grid-cols-[minmax(0,1fr)_auto] lg:grid-cols-[minmax(0,1fr)_180px_auto_104px_auto]"
                >
                  <div className="min-w-0">
                    <div className="font-medium text-sm truncate">{b.label || b.client_id}</div>
                    <div className="text-xs text-muted-foreground truncate">
                      <code className="text-[11px]">{b.client_id}</code>
                      {' · '}
                      {b.last_seen_at ? (
                        <span
                          title={t('settings.mcp.bindings.lastSeenTitle', { when: b.last_seen_at })}
                        >
                          <Plug className="inline h-3 w-3 text-emerald-500" />{' '}
                          {t('settings.mcp.bindings.lastSeen', {
                            when: formatDate(b.last_seen_at),
                          })}
                        </span>
                      ) : (
                        <span>{t('settings.mcp.bindings.neverConnected')}</span>
                      )}
                    </div>
                  </div>
                  <Select
                    value={b.profile_id ?? '__default__'}
                    onValueChange={(v) =>
                      saveBinding(b, { profile_id: v === '__default__' ? null : v })
                    }
                  >
                    <SelectTrigger className="w-full sm:w-[180px]">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="__default__">
                        {t('settings.mcp.bindings.defaultOption')}
                      </SelectItem>
                      {(profiles ?? []).map((p) => (
                        <SelectItem key={p.id} value={p.id}>
                          {p.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <label
                    htmlFor={`mcp-plain-text-${b.client_id}`}
                    className="flex items-center gap-2 text-xs text-muted-foreground whitespace-nowrap cursor-pointer"
                    title={t('settings.mcp.bindings.plainTextTitle')}
                  >
                    <Checkbox
                      id={`mcp-plain-text-${b.client_id}`}
                      checked={b.default_plain_text}
                      onCheckedChange={(v) => saveBinding(b, { default_plain_text: v })}
                    />
                    {t('settings.mcp.bindings.plainText')}
                  </label>
                  <Input
                    // Uncontrolled on purpose: commit on blur / Enter, not per keystroke.
                    key={`${b.client_id}:${b.default_max_chars ?? ''}`}
                    type="number"
                    min={50}
                    max={10000}
                    step={50}
                    defaultValue={b.default_max_chars ?? ''}
                    placeholder={t('settings.mcp.bindings.maxCharsPlaceholder')}
                    title={t('settings.mcp.bindings.maxCharsTitle')}
                    className="h-9 w-full px-2 text-xs sm:w-[104px]"
                    onBlur={(e) => commitMaxChars(b, e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') e.currentTarget.blur();
                    }}
                  />
                  <Button
                    size="icon"
                    variant="ghost"
                    onClick={() => remove(b.client_id)}
                    aria-label={t('settings.mcp.bindings.removeAria', { client: b.client_id })}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              ))}
            </div>
          )}

          <div className="pt-4 space-y-2">
            <div className="text-sm font-medium">{t('settings.mcp.bindings.add.title')}</div>
            <div className="grid grid-cols-[1fr_1fr_auto] gap-2">
              <input
                type="text"
                placeholder={t('settings.mcp.bindings.add.clientIdPlaceholder')}
                value={newClientId}
                onChange={(e) => setNewClientId(e.target.value)}
                className="h-9 px-3 rounded-md border bg-background text-sm"
              />
              <input
                type="text"
                placeholder={t('settings.mcp.bindings.add.labelPlaceholder')}
                value={newLabel}
                onChange={(e) => setNewLabel(e.target.value)}
                className="h-9 px-3 rounded-md border bg-background text-sm"
              />
              <Select
                value={newProfileId || '__default__'}
                onValueChange={(v) => setNewProfileId(v === '__default__' ? '' : v)}
              >
                <SelectTrigger className="h-9 min-w-[140px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__default__">
                    {t('settings.mcp.bindings.defaultOption')}
                  </SelectItem>
                  {(profiles ?? []).map((p) => (
                    <SelectItem key={p.id} value={p.id}>
                      {p.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <Button size="sm" onClick={handleAdd} disabled={!newClientId.trim() || adding}>
              {t('settings.mcp.bindings.add.action')}
            </Button>
          </div>
        </SettingSection>
      </div>

      <aside className="hidden lg:block w-[280px] shrink-0 space-y-6 sticky top-0">
        <div className="space-y-2">
          <h3 className="text-sm font-semibold">{t('settings.mcp.sidebar.aboutTitle')}</h3>
          <p className="text-sm text-muted-foreground leading-relaxed">
            {t('settings.mcp.sidebar.aboutBody')}
          </p>
        </div>
        <div className="space-y-2">
          <h3 className="text-sm font-semibold">{t('settings.mcp.sidebar.toolsTitle')}</h3>
          <ul className="text-sm text-muted-foreground space-y-1.5 leading-relaxed">
            <li>
              <code className="text-accent">voicebox.speak</code>
              <div>{t('settings.mcp.sidebar.tools.speak')}</div>
            </li>
            <li>
              <code className="text-accent">voicebox.transcribe</code>
              <div>{t('settings.mcp.sidebar.tools.transcribe')}</div>
            </li>
            <li>
              <code className="text-accent">voicebox.list_captures</code>
              <div>{t('settings.mcp.sidebar.tools.listCaptures')}</div>
            </li>
            <li>
              <code className="text-accent">voicebox.list_profiles</code>
              <div>{t('settings.mcp.sidebar.tools.listProfiles')}</div>
            </li>
            <li>
              <code className="text-accent">voicebox.speech_*</code>
              <div>{t('settings.mcp.sidebar.tools.speechSessions')}</div>
            </li>
          </ul>
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Radio className="h-3.5 w-3.5 text-accent" />
          <span>
            <Trans i18nKey="settings.mcp.sidebar.agentBridge" components={{ code: <code /> }} />
          </span>
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Terminal className="h-3.5 w-3.5 text-accent" />
          <span>
            <Trans i18nKey="settings.mcp.sidebar.postSpeak" components={{ code: <code /> }} />
          </span>
        </div>
      </aside>
    </div>
  );
}

function ToolList({ tools }: { tools: readonly string[] }) {
  return (
    <div className="flex flex-wrap gap-2">
      {tools.map((tool) => (
        <code key={tool} className="text-[11px] rounded-md border bg-background px-2 py-1">
          {tool}
        </code>
      ))}
    </div>
  );
}

function SnippetRow({
  title,
  description,
  snippet,
}: {
  title: string;
  description: string;
  snippet: string;
}) {
  const { t } = useTranslation();
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(snippet);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // ignore; user can still select-and-copy the pre content
    }
  };

  return (
    <div className="py-3 space-y-2">
      <div className="flex items-center justify-between gap-4">
        <div>
          <div className="text-sm font-medium">{title}</div>
          <div className="text-xs text-muted-foreground">{description}</div>
        </div>
        <Button size="sm" variant="outline" onClick={copy}>
          {copied ? (
            <>
              <Check className="h-3.5 w-3.5 mr-1.5" />
              {t('settings.mcp.install.copied')}
            </>
          ) : (
            <>
              <Copy className="h-3.5 w-3.5 mr-1.5" />
              {t('settings.mcp.install.copy')}
            </>
          )}
        </Button>
      </div>
      <pre className="text-[11px] font-mono p-3 rounded-md bg-muted/50 overflow-x-auto whitespace-pre-wrap break-all">
        {snippet}
      </pre>
    </div>
  );
}
