import { describe, expect, test } from 'bun:test';
import ts from 'typescript';
import { patchSettings } from '../../scripts/install-agent-bridge.mjs';

describe('user settings JSONC edits', () => {
  for (const original of ['{}', '{"x":1}', '{"x":1,}', '{\n// comment\n"x":1 // retained\n}']) {
    test(`adds wrapper keys without rewriting existing settings: ${original}`, () => {
      const result = patchSettings(original, {
        'claudeCode.claudeProcessWrapper': '/some path/wrapper',
      });
      const parsed = ts.parseConfigFileTextToJson('settings.json', result);
      expect(parsed.error).toBeUndefined();
      expect(parsed.config['claudeCode.claudeProcessWrapper']).toBe('/some path/wrapper');
      if (original.includes('// retained')) expect(result).toContain('// retained');
      if (original.includes('"x":1')) expect(result).toContain('"x":1');
    });
  }
  test('replaces only the target value, preserving comments and unrelated keys', () => {
    const result = patchSettings('{/*before*/"chatgpt.cliExecutable":null, "other":true}', {
      'chatgpt.cliExecutable': '/wrapper',
    });
    expect(result).toBe('{/*before*/"chatgpt.cliExecutable":"/wrapper", "other":true}');
  });
  test('refuses malformed or ambiguous configuration', () => {
    expect(() => patchSettings('{broken', { x: 1 })).toThrow();
    expect(() => patchSettings('{"x":1,"x":2}', { x: 3 })).toThrow();
  });
});
