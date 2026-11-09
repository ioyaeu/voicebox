/**
 * Platform abstraction types
 * These interfaces define the contract that platform implementations must fulfill
 */

export interface FileFilter {
  name: string;
  extensions: string[];
}

export interface PlatformFilesystem {
  saveFile(filename: string, blob: Blob, filters?: FileFilter[]): Promise<void>;
  openPath(path: string): Promise<void>;
  pickDirectory(title: string): Promise<string | null>;
}

export interface UpdateStatus {
  checking: boolean;
  available: boolean;
  version?: string;
  downloading: boolean;
  installing: boolean;
  readyToInstall: boolean;
  error?: string;
  downloadProgress?: number; // 0-100 percentage
  downloadedBytes?: number;
  totalBytes?: number;
}

export interface PlatformUpdater {
  checkForUpdates(): Promise<void>;
  downloadAndInstall(): Promise<void>;
  restartAndInstall(): Promise<void>;
  getStatus(): UpdateStatus;
  subscribe(callback: (status: UpdateStatus) => void): () => void;
}

export interface AudioDevice {
  id: string;
  name: string;
  is_default: boolean;
}

export interface PlatformAudio {
  isSystemAudioSupported(): Promise<boolean>;
  startSystemAudioCapture(maxDurationSecs: number): Promise<void>;
  stopSystemAudioCapture(): Promise<Blob>;
  listOutputDevices(): Promise<AudioDevice[]>;
  playToDevices(audioData: Uint8Array, deviceIds: string[]): Promise<void>;
  stopPlayback(): void;
}

export interface ServerLogEntry {
  stream: 'stdout' | 'stderr';
  line: string;
}

export interface PlatformLifecycle {
  startServer(remote?: boolean, modelsDir?: string | null): Promise<string>;
  stopServer(): Promise<void>;
  /** `remote` overrides the network mode; omitted, the network-access setting applies. */
  restartServer(modelsDir?: string | null, remote?: boolean): Promise<string>;
  /** True when this app controls the server process, so stop/restart act on it. */
  isServerManaged(): Promise<boolean>;
  setKeepServerRunning(keep: boolean): Promise<void>;
  setBackendOverride(backend?: string | null): Promise<void>;
  setupWindowCloseHandler(): Promise<void>;
  subscribeToServerLogs(callback: (entry: ServerLogEntry) => void): () => void;
  onServerReady?: () => void;
}

export interface PlatformMetadata {
  getVersion(): Promise<string>;
  isTauri: boolean;
}

export interface Platform {
  filesystem: PlatformFilesystem;
  updater: PlatformUpdater;
  audio: PlatformAudio;
  lifecycle: PlatformLifecycle;
  metadata: PlatformMetadata;
}
