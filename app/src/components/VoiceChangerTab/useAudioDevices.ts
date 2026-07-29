import { useCallback, useEffect, useState } from 'react';

export interface AudioDeviceOption {
  deviceId: string;
  label: string;
}

export type VirtualCableOS = 'macos' | 'windows' | 'other';

export interface UseAudioDevicesResult {
  inputs: AudioDeviceOption[];
  outputs: AudioDeviceOption[];
  /** Output device that looks like a virtual audio cable, if one is installed. */
  virtualOutput: AudioDeviceOption | null;
  hasVirtualOutput: boolean;
  /** Whether `HTMLMediaElement.setSinkId` (output routing) exists in this webview. */
  outputRoutingSupported: boolean;
  /** Best-effort OS detection, used only to pick the right setup instructions. */
  os: VirtualCableOS;
  /** Labels are only populated after mic permission is granted. */
  labelsAvailable: boolean;
  refresh: () => Promise<void>;
  /**
   * Runs a `getUserMedia({audio:true})` permission preflight (stopping the
   * tracks immediately) then re-enumerates so device labels become available
   * without starting a conversion. Resolves once the refresh completes; a
   * denied/failed prompt leaves labels hidden and is swallowed (the CTA stays).
   */
  requestPermission: () => Promise<void>;
}

// BlackHole (macOS) and VB-Audio VB-CABLE / Voicemeeter (Windows) are the
// virtual devices the setup guidance targets.
const VIRTUAL_OUTPUT_PATTERN = /blackhole|vb-?cable|vb-?audio|voicemeeter|cable input/i;

function detectOs(): VirtualCableOS {
  if (typeof navigator === 'undefined') return 'other';
  const ua = `${navigator.userAgent} ${navigator.platform ?? ''}`.toLowerCase();
  if (ua.includes('mac')) return 'macos';
  if (ua.includes('win')) return 'windows';
  return 'other';
}

function isSinkIdSupported(): boolean {
  return (
    typeof HTMLMediaElement !== 'undefined' &&
    typeof HTMLMediaElement.prototype.setSinkId === 'function'
  );
}

/**
 * Enumerates audio input/output devices for the real-time voice changer and
 * flags whether a virtual output cable is installed. Device labels only appear
 * once microphone permission has been granted, so callers should `refresh()`
 * after starting a stream.
 */
export function useAudioDevices(): UseAudioDevicesResult {
  const [inputs, setInputs] = useState<AudioDeviceOption[]>([]);
  const [outputs, setOutputs] = useState<AudioDeviceOption[]>([]);
  const [labelsAvailable, setLabelsAvailable] = useState(false);
  const os = detectOs();
  const outputRoutingSupported = isSinkIdSupported();

  const refresh = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) {
      return;
    }
    const devices = await navigator.mediaDevices.enumerateDevices();
    const nextInputs: AudioDeviceOption[] = [];
    const nextOutputs: AudioDeviceOption[] = [];
    let anyLabel = false;
    for (const device of devices) {
      if (device.label) anyLabel = true;
      const option: AudioDeviceOption = {
        deviceId: device.deviceId,
        label: device.label || device.deviceId || 'Unknown device',
      };
      if (device.kind === 'audioinput') {
        nextInputs.push(option);
      } else if (device.kind === 'audiooutput') {
        nextOutputs.push(option);
      }
    }
    setInputs(nextInputs);
    setOutputs(nextOutputs);
    setLabelsAvailable(anyLabel);
  }, []);

  const requestPermission = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      return;
    }
    try {
      // Preflight only: opening the stream unlocks device labels for the rest of
      // the session. Stop the tracks immediately so no mic is held open.
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      for (const track of stream.getTracks()) {
        track.stop();
      }
    } catch {
      // Denied or unavailable: labels stay hidden and the grant CTA remains.
      // Still refresh below in case a prior grant already populated labels.
    }
    await refresh();
  }, [refresh]);

  useEffect(() => {
    void refresh();
    if (!navigator.mediaDevices) return;
    const handler = () => void refresh();
    navigator.mediaDevices.addEventListener('devicechange', handler);
    return () => {
      navigator.mediaDevices.removeEventListener('devicechange', handler);
    };
  }, [refresh]);

  const virtualOutput = outputs.find((o) => VIRTUAL_OUTPUT_PATTERN.test(o.label)) ?? null;

  return {
    inputs,
    outputs,
    virtualOutput,
    hasVirtualOutput: virtualOutput !== null,
    outputRoutingSupported,
    os,
    labelsAvailable,
    refresh,
    requestPermission,
  };
}
