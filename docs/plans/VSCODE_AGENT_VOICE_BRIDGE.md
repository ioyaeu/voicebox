# VS Code Agent Voice Bridge

Status: opt-in local macOS prototype. This is the host-adapter follow-up to
[MCP speech sessions](MCP_SPEECH_STREAMING.md), not a replacement chat UI.

## Integration Points

- Claude: the documented `claudeCode.claudeProcessWrapper` receives the bundled
  executable followed by its arguments. The installed local extension uses
  bidirectional `stream-json` with partial messages. The relay does not add or
  remove CLI flags, tool permissions, approval responses or prompts.
- Codex: `chatgpt.cliExecutable` is explicitly marked DEVELOPMENT ONLY in the
  installed extension. Opting into this experimental setting launches the
  relay instead of the bundled executable. The real App Server is still used,
  with its stdio JSONL protocol. Non-stdio transports are passed through
  without narration. The extension registry is consulted on launch to follow
  ordinary Codex extension upgrades; that does not guarantee protocol compatibility.

Sources checked September 2026:
[Claude VS Code settings](https://code.claude.com/docs/en/vs-code#extension-settings),
[Claude stream events](https://code.claude.com/docs/en/agent-sdk/streaming-output),
[Codex App Server](https://learn.chatgpt.com/docs/app-server).
Local extension inspection: Claude 2.1.263 and Codex extension 26.901.22334
(bundled CLI 0.153.0). No extension files were modified.

The wrapper can change Claude's initial permission-mode resolution: the
documented default in a wrapped setup is Manual unless an explicit setting or
previous UI choice overrides it. This installer does not compensate by
enabling auto-accept or bypass mode. Review the mode after reloading VS Code.

## Architecture

```text
VS Code stdin -> relay -> actual agent process
VS Code stdout <- relay <- actual agent process
                    |
          bounded in-memory event mirror
                    |
          public text filter and turn lifecycle
                    |
          bounded background audio consumer
                    |
          local Voicebox speech-session REST API
```

The IDE pipes are byte-preserving; stderr and exit codes are preserved. Normal
termination signals are forwarded to the direct agent process. No stdin or
stdout content is logged to disk. The mirror queue is bounded to 128 chunks of
8 KiB; JSON records are limited to 1 MiB. Overflow disables narration while
the original protocol continues. The audio consumer keeps at most 64 KiB of
unconsumed text; overflow cancels that turn's narration rather than blocking
the IDE or silently dropping words from a continuing speech session.

Only public assistant text is selected. Codex requires live thread/turn/item
identity and the `agentMessage` type. Claude selects root-session text blocks,
deduplicates the subsequent complete message, and supports whole-message
fallback when partial events are absent. Reasoning, tools, tool results,
subagents, approval requests and session history emitted before a live user
request are not narrated. Public progress comments are included: this mode is
not restricted to the final answer.

Markdown filtering reuses Voicebox's prose cleanup, with bounded line/fence
state across deltas. Plain prose can flush on sentence boundaries. Marked-up
lines wait until newline or message completion; code fences, indented code,
table-like lines and oversized lines are discarded. This is conservative
speech preparation, not a complete incremental CommonMark implementation.

New user requests and interrupts cancel the previous narration. Claude queued
requests wait for the preceding result before arming the next response, so
late output is not mistaken for the new answer. When an explicit Voicebox
`speak` or `speech_start` tool is observed, the bridge yields the rest of that
turn to MCP. Do not configure automatic MCP narration or a Stop speech hook
at the same time: a direct tool call can still race cancellation/cleanup and
receive a busy error. Explicit audio creation through MCP remains available.

Voicebox retains authority over voice, engine, playback ownership and file
lifetime. The bridge uses REST because it already has the public text events;
it does not ask the language model to issue MCP calls. The REST and MCP session
surfaces share the same service. Client ids must match existing Voicebox
bindings. Defaults are `codex-vscode` and `claude-code`; no voice is hardcoded.

## Install And Restore

Build Voicebox with the speech-session endpoints and the corresponding Tauri
player first. An older server exposing only `/speak` cannot play the bridge's
sessions. The relay remains usable for IDE traffic in that case, logging only
an audio failure category to stderr.

From the repository root, preview installation:

```sh
node scripts/install-agent-bridge.mjs --claude --codex-experimental
```

Explicitly opt into installation:

```sh
node scripts/install-agent-bridge.mjs --claude --codex-experimental --apply
```

The installer copies a self-contained standard-library Python runtime into
`~/.local/share/voicebox/agent-bridge/`. Switching repository branches cannot
remove the installed relay. Python 3.9+ is required. Node and the repository's
TypeScript dependency are needed only for installation and JSONC-aware edits.

Only two VS Code user settings are changed, preserving other values and
comments. Existing third-party wrappers are refused. An installation manifest
stores the original settings locally with mode 0600; it can contain sensitive
configuration and must never be committed or shared. The runtime's
`config.json` contains the loopback URL, client ids and an `enabled` toggle.
No agent instruction file, MCP config or permission setting is changed.

After installation, reload VS Code to use the wrappers for new agent processes.
Do not reload while an agent is editing files. Set `enabled` to false in the
runtime config to launch future processes without narration, or remove the
two wrapper settings. For exact rollback when settings have not since changed:

```sh
node scripts/install-agent-bridge.mjs --restore --apply
```

Rollback refuses to replace a settings file modified since installation.
In that case remove only `claudeCode.claudeProcessWrapper` and
`chatgpt.cliExecutable`; retain unrelated user edits. Reinstallation also
refuses to overwrite a stale rollback snapshot. Runtime files are retained.

## Audio Retention And Failure

For ordinary responses, audio and generation rows are temporary. A user line
beginning with `keep audio` sets retention for that response, for example:

```text
keep audio, explique-moi les changements de cette branche.
```

Quoted occurrences elsewhere, agent output and tool results cannot switch
retention on. Voicebox stores kept phrases separately, not as a joined export.

HTTP traffic is restricted to loopback without environment proxies or
redirects. No credentials or cloud endpoints are introduced. Only bounded
public prose leaves the relay for Voicebox. Append retries reuse sequence and
text; the start request is not blindly retried after a lost response. A lost
start reply may leave an orphan until the server's renderer lease expires.

Voicebox currently permits only one global speech session. A second busy agent
is not queued indefinitely: its narration fails for that turn, without
interrupting the first agent's audio or its own IDE conversation. Playback
stop, backend failure or queue overflow likewise silence the remainder of
that turn; a new user request can try again. Pausing the Voicebox pill does not
pause the agent; a sufficiently long pause can fill the bounded text queue.

Known limits: the server's five-minute producer-idle timeout still applies
during long tool-only gaps; periodic status polling is not a producer lease
renewal. A session that expires is not automatically recreated mid-answer.
Closing the agent process cancels its audio, so this wrapper is intended for
persistent IDE sessions, not a one-shot CLI that exits immediately after output.
Remote VS Code, Windows, Linux and simultaneous multi-agent audio are not
validated by this installer. Do not promote the Codex override as a supported
extension API.

## Verification And Next Gate

```sh
backend/venv/bin/python -m pytest backend/tests/test_agent_bridge.py backend/tests/test_agent_bridge_wire.py -q
bun test app/tests/agentBridgeInstaller.test.ts
```

Tests cover both event formats, hidden/non-root filtering, message deduplication,
queued requests, Markdown fragmentation, bounded memory, interrupted slow
starts, retry-safe sequences, retained audio intent, JSONC settings, executable
discovery and byte-preserving real subprocess pipes. The wire test binds an
ephemeral loopback port with a fake server and synthetic agent, never a model.
Installed-wrapper version checks and the real Codex `initialize` handshake
were also verified without starting a conversation or calling a model.

Release gate: after rebuild/reload, test a real conversation in each original
VS Code panel, including permission prompts, a long answer, pause/resume,
interrupt, queued follow-up, a Markdown code block, keep-audio and Voicebox
unavailable. Verify exactly one reader and no ephemeral History leftovers.
Synthetic tests and an initialize handshake do not establish that listening
experience. Next hardening: explicit producer leases for long tool gaps,
packaged consent/installation UI, more protocol-version fixtures, and a
supported Codex host integration to replace the development override.
