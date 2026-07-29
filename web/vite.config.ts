import path from 'node:path';
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import { changelogPlugin } from '../app/plugins/changelog';

export default defineConfig({
  plugins: [react(), tailwindcss(), changelogPlugin(path.resolve(__dirname, '..'))],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, '../app/src'),
    },
  },
  build: {
    outDir: 'dist',
    // Keep the realtime RVC AudioWorklet modules as physical hashed assets
    // instead of inlined base64 `data:` URIs (see app/vite.config.ts for the
    // full rationale). The web build shares app/src via the `@` alias, so the
    // same override is required here. `undefined` for other assets keeps the
    // default inline behavior byte-for-byte.
    assetsInlineLimit: (filePath: string) =>
      /[\\/]worklets[\\/]/.test(filePath) ? false : undefined,
  },
});
