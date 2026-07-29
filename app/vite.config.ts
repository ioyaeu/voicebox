import path from 'node:path';
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import { changelogPlugin } from './plugins/changelog';

export default defineConfig({
  plugins: [tailwindcss(), react(), changelogPlugin(path.resolve(__dirname, '..'))],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    // The realtime RVC AudioWorklet modules are referenced via
    // `new URL('../worklets/*.js', import.meta.url)` so Vite emits them as
    // hashed assets. `AudioWorklet.addModule()` must fetch a real script URL,
    // and inlining a worklet as a base64 `data:` URI is both unreliable on
    // WebKit (Tauri macOS webview) and invisible as a build artifact. The small
    // capture worklet is under Vite's 4 kB `assetsInlineLimit` and would
    // otherwise inline — force these two files to stay physical. Returning
    // `undefined` for everything else preserves the default inline behavior
    // byte-for-byte, so no non-worklet asset changes.
    assetsInlineLimit: (filePath: string) =>
      /[\\/]worklets[\\/]/.test(filePath) ? false : undefined,
  },
});
