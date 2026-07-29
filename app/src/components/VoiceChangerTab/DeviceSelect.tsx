import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import type { AudioDeviceOption } from './useAudioDevices';

// Radix Select cannot hold an empty-string item value; this sentinel represents
// "system default" and is mapped back to `undefined` by the panel.
export const DEFAULT_DEVICE_VALUE = '__default__';

// Chromium/WebView2 returns deviceId "" for every device before mic permission
// is granted. Radix Select throws on an empty item value, so those devices are
// collapsed into this single disabled placeholder instead of being rendered.
const NO_PERMISSION_VALUE = '__no_permission__';

interface DeviceSelectProps {
  id: string;
  label: string;
  defaultOptionLabel: string;
  devices: AudioDeviceOption[];
  value: string;
  onValueChange: (value: string) => void;
  disabled?: boolean;
  placeholder: string;
}

export function DeviceSelect({
  id,
  label,
  defaultOptionLabel,
  devices,
  value,
  onValueChange,
  disabled = false,
  placeholder,
}: DeviceSelectProps) {
  const selectableDevices = devices.filter((device) => device.deviceId !== '');
  const hasBlockedDevices = selectableDevices.length < devices.length;
  return (
    <div className="space-y-1.5">
      <Label htmlFor={id}>{label}</Label>
      <Select value={value} onValueChange={onValueChange} disabled={disabled}>
        <SelectTrigger id={id}>
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={DEFAULT_DEVICE_VALUE}>{defaultOptionLabel}</SelectItem>
          {selectableDevices.map((device) => (
            <SelectItem key={device.deviceId} value={device.deviceId}>
              {device.label}
            </SelectItem>
          ))}
          {hasBlockedDevices && (
            <SelectItem value={NO_PERMISSION_VALUE} disabled>
              Grant microphone access to list devices
            </SelectItem>
          )}
        </SelectContent>
      </Select>
    </div>
  );
}
