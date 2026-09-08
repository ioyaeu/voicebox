#!/usr/bin/env node
// Installation is explicit and per-user. VS Code extension files are read only.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';

export function patchSettings(text, updates) {
  const source = ts.parseJsonText('settings.json', text);
  if (source.parseDiagnostics.length) throw new Error('Invalid VS Code settings JSONC');
  const object = source.statements[0]?.expression;
  if (!object || !ts.isObjectLiteralExpression(object))
    throw new Error('Settings must be an object');
  const edits = [];
  const additions = [];
  for (const [key, value] of Object.entries(updates)) {
    const matches = object.properties.filter((p) => p.name?.text === key);
    if (matches.length > 1) throw new Error(`Duplicate setting: ${key}`);
    if (matches.length) {
      const node = matches[0].initializer;
      edits.push({ start: node.getStart(source), end: node.end, text: JSON.stringify(value) });
    } else additions.push(`  ${JSON.stringify(key)}: ${JSON.stringify(value)}`);
  }
  if (additions.length) {
    const last = object.properties.at(-1);
    if (last && !object.properties.hasTrailingComma)
      edits.push({ start: last.end, end: last.end, text: ',' });
    edits.push({
      start: object.end - 1,
      end: object.end - 1,
      text: `\n${additions.join(',\n')}\n`,
    });
  }
  // Later insertions at an identical offset must be applied first.
  for (const edit of edits.reverse().sort((a, b) => b.start - a.start))
    text = text.slice(0, edit.start) + edit.text + text.slice(edit.end);
  const check = ts.parseConfigFileTextToJson('settings.json', text);
  if (check.error) throw new Error('Could not safely update settings');
  return text;
}

const hash = (value) => crypto.createHash('sha256').update(value).digest('hex');

function main() {
  const args = new Set(process.argv.slice(2));
  if (process.platform !== 'darwin')
    throw new Error('This installer currently targets local macOS VS Code only');
  const root = path.join(os.homedir(), '.local/share/voicebox/agent-bridge');
  const settingsPath = path.join(
    os.homedir(),
    'Library/Application Support/Code/User/settings.json',
  );
  const original = fs.readFileSync(settingsPath, 'utf8');
  const manifestPath = path.join(root, 'installation.json');
  if (args.has('--restore')) {
    const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    if (hash(original) !== manifest.installedHash)
      throw new Error(
        'Settings changed since installation; restore only the two wrapper keys manually, not the whole backup',
      );
    if (!args.has('--apply')) {
      console.log('Dry run: would restore previous VS Code settings');
      return;
    }
    fs.writeFileSync(settingsPath, manifest.original, { mode: 0o600 });
    console.log('Previous VS Code settings restored. Reload VS Code. Runtime files retained.');
    return;
  }
  const updates = {};
  if (args.has('--claude'))
    updates['claudeCode.claudeProcessWrapper'] = path.join(root, 'voicebox-claude');
  if (args.has('--codex-experimental'))
    updates['chatgpt.cliExecutable'] = path.join(root, 'voicebox-codex');
  if (!Object.keys(updates).length)
    throw new Error(
      'Choose --claude and/or --codex-experimental; --apply writes, --restore rolls back',
    );
  const parsed = ts.parseConfigFileTextToJson(settingsPath, original);
  if (parsed.error) throw new Error('Cannot read VS Code settings');
  for (const [key, value] of Object.entries(updates)) {
    if (parsed.config[key] != null && parsed.config[key] !== value)
      throw new Error(`Existing ${key} is not ours; refusing to replace it`);
  }
  const result = patchSettings(original, updates);
  const prior = fs.existsSync(manifestPath)
    ? JSON.parse(fs.readFileSync(manifestPath, 'utf8'))
    : null;
  const alreadyInstalled = Object.entries(updates).some(
    ([key, value]) => parsed.config[key] === value,
  );
  if (prior && alreadyInstalled && hash(original) !== prior.installedHash) {
    throw new Error(
      'Settings changed since installation; refusing to replace the rollback snapshot. Remove the wrapper keys before reinstalling.',
    );
  }
  const extensionsRoot = path.join(os.homedir(), '.vscode/extensions');
  const extensions = JSON.parse(
    fs.readFileSync(path.join(extensionsRoot, 'extensions.json'), 'utf8'),
  );
  const codex = extensions.find((e) => e.identifier.id === 'openai.chatgpt');
  const relative = process.arch === 'arm64' ? 'bin/macos-aarch64/codex' : 'bin/macos-x86_64/codex';
  const binary = codex ? path.join(codex.location.path, relative) : null;
  if (args.has('--codex-experimental') && (!binary || !fs.existsSync(binary)))
    throw new Error('Bundled Codex executable not found');
  const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
  const files = [
    'backend/__init__.py',
    'backend/utils/__init__.py',
    'backend/utils/speech_text.py',
    ...fs
      .readdirSync(path.join(repo, 'backend/agent_bridge'))
      .filter((f) => f.endsWith('.py'))
      .map((f) => `backend/agent_bridge/${f}`),
  ];
  const python = execFileSync('which', ['python3'], { encoding: 'utf8' }).trim();
  const version = execFileSync(
    python,
    ['-c', 'import sys; print(sys.version_info.major, sys.version_info.minor)'],
    { encoding: 'utf8' },
  )
    .trim()
    .split(' ')
    .map(Number);
  if (version[0] !== 3 || version[1] < 9) throw new Error('Python 3.9 or newer is required');
  console.log(
    JSON.stringify(
      {
        apply: args.has('--apply'),
        settingsPath,
        updates,
        runtime: root,
        python,
        note: 'Codex override is experimental. Claude wrapper may change default permission resolution to Manual. No permission bypass is added.',
      },
      null,
      2,
    ),
  );
  if (!args.has('--apply')) return;
  fs.mkdirSync(root, { recursive: true, mode: 0o700 });
  // Keep a recoverable copy before any settings change. It contains user settings
  // and must never be checked into the repo or printed to logs.
  const manifest = {
    original: alreadyInstalled ? (prior?.original ?? original) : original,
    installedHash: hash(result),
    installedAt: new Date().toISOString(),
  };
  fs.writeFileSync(manifestPath, JSON.stringify(manifest), { mode: 0o600 });
  for (const file of files) {
    const target = path.join(root, file);
    fs.mkdirSync(path.dirname(target), { recursive: true, mode: 0o700 });
    fs.copyFileSync(path.join(repo, file), target);
  }
  const config = {
    enabled: true,
    url: 'http://127.0.0.1:17493',
    clients: { codex: 'codex-vscode', claude: 'claude-code' },
    extensions_root: extensionsRoot,
    codex_relative_binary: relative,
    codex_executable: binary,
  };
  const configPath = path.join(root, 'config.json');
  if (!fs.existsSync(configPath))
    fs.writeFileSync(configPath, JSON.stringify(config, null, 2), { mode: 0o600 });
  for (const protocol of ['claude', 'codex']) {
    const launcher = `#!${python}\nimport sys\nfrom pathlib import Path\nfrom backend.agent_bridge.proxy import main\nif __name__ == '__main__':\n    sys.exit(main(${JSON.stringify(protocol)}, sys.argv[1:], Path(__file__).with_name('config.json')))\n`;
    fs.writeFileSync(path.join(root, `voicebox-${protocol}`), launcher, { mode: 0o700 });
  }
  // Detect a concurrent VS Code settings write instead of overwriting it.
  if (fs.readFileSync(settingsPath, 'utf8') !== original)
    throw new Error('Settings changed during installation; rerun');
  fs.writeFileSync(settingsPath, result);
  console.log('Installed. Reload VS Code after rebuilding Voicebox with speech-session endpoints.');
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    main();
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
