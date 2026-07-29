import { AlertCircle, FileAudio, Upload, X } from 'lucide-react';
import { useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils/cn';

const AUDIO_ACCEPT = 'audio/*,.wav,.mp3,.m4a,.flac,.ogg,.aac,.opus';
// Kept in sync with the extensions listed in AUDIO_ACCEPT above — a dropped file
// with no MIME type is matched against exactly the same set the picker accepts.
const AUDIO_EXTENSIONS = /\.(wav|mp3|m4a|flac|ogg|aac|opus)$/i;

function isAudioFile(file: File): boolean {
  if (file.type.startsWith('audio/')) return true;
  return AUDIO_EXTENSIONS.test(file.name);
}

interface SourceFilePickerProps {
  file: File | null;
  onSelect: (file: File) => void;
  onClear: () => void;
  disabled?: boolean;
}

/** Drop zone + file picker for the source recording that gets converted. */
export function SourceFilePicker({ file, onSelect, onClear, disabled = false }: SourceFilePickerProps) {
  const { t } = useTranslation();
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function handleFiles(files: FileList | null) {
    const selected = files?.[0];
    if (!selected) return;
    if (isAudioFile(selected)) {
      setError(null);
      onSelect(selected);
    } else {
      // Surface the rejection instead of silently swallowing the file.
      setError(t('voiceChanger.source.invalidFile'));
    }
  }

  if (file) {
    return (
      <div className="flex items-center gap-2 rounded-md border border-border bg-muted/40 px-3 py-2.5">
        <FileAudio className="h-4 w-4 shrink-0 text-accent" />
        <span className="flex-1 truncate text-sm">{file.name}</span>
        <Button
          type="button"
          size="icon"
          variant="ghost"
          className="h-7 w-7"
          disabled={disabled}
          onClick={onClear}
          aria-label={t('voiceChanger.source.clear')}
        >
          <X className="h-4 w-4" />
        </Button>
      </div>
    );
  }

  return (
    <>
      <input
        type="file"
        accept={AUDIO_ACCEPT}
        ref={inputRef}
        disabled={disabled}
        className="hidden"
        onChange={(e) => {
          handleFiles(e.target.files);
          e.target.value = '';
        }}
      />
      <button
        type="button"
        disabled={disabled}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragging(false);
          if (!disabled) handleFiles(e.dataTransfer.files);
        }}
        className={cn(
          'flex w-full flex-col items-center justify-center gap-2 rounded-lg border border-dashed px-4 py-8 text-center transition-colors',
          'disabled:cursor-not-allowed disabled:opacity-50',
          isDragging
            ? 'border-accent bg-accent/10'
            : 'border-border hover:border-accent/60 hover:bg-muted/40',
        )}
      >
        <Upload className="h-6 w-6 text-muted-foreground" />
        <span className="text-sm font-medium">{t('voiceChanger.source.dropTitle')}</span>
        <span className="text-xs text-muted-foreground">{t('voiceChanger.source.dropHint')}</span>
      </button>
      {error && (
        <div className="mt-2 flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
          <span className="break-words">{error}</span>
        </div>
      )}
    </>
  );
}
