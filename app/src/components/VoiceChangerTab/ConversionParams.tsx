import { ChevronDown, ChevronUp, SlidersHorizontal } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import type { RvcF0Method } from '@/lib/api/types';

interface ConversionParamsProps {
  f0UpKey: number;
  onF0UpKey: (value: number) => void;
  f0Method: RvcF0Method;
  onF0Method: (value: RvcF0Method) => void;
  indexRate: number;
  onIndexRate: (value: number) => void;
  rmsMixRate: number;
  onRmsMixRate: (value: number) => void;
  protect: number;
  onProtect: (value: number) => void;
  /** Whether the selected profile has a feature index — index rate is inert without one. */
  hasIndex: boolean;
  disabled?: boolean;
}

function formatSemitones(value: number): string {
  return value > 0 ? `+${value}` : String(value);
}

export function ConversionParams({
  f0UpKey,
  onF0UpKey,
  f0Method,
  onF0Method,
  indexRate,
  onIndexRate,
  rmsMixRate,
  onRmsMixRate,
  protect,
  onProtect,
  hasIndex,
  disabled = false,
}: ConversionParamsProps) {
  const { t } = useTranslation();
  const [advancedOpen, setAdvancedOpen] = useState(false);

  return (
    <div className="space-y-5">
      {/* Prominent pitch control */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label htmlFor="rvc-pitch">{t('voiceChanger.params.pitch.label')}</Label>
          <span className="text-sm font-medium tabular-nums text-muted-foreground">
            {t('voiceChanger.params.pitch.value', { value: formatSemitones(f0UpKey) })}
          </span>
        </div>
        <Slider
          id="rvc-pitch"
          value={[f0UpKey]}
          onValueChange={([value]) => onF0UpKey(value)}
          min={-24}
          max={24}
          step={1}
          disabled={disabled}
          aria-label={t('voiceChanger.params.pitch.label')}
        />
        <p className="text-xs text-muted-foreground">{t('voiceChanger.params.pitch.hint')}</p>
      </div>

      <button
        type="button"
        onClick={() => setAdvancedOpen((open) => !open)}
        className="flex items-center gap-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground"
      >
        <SlidersHorizontal className="h-4 w-4" />
        {t('voiceChanger.params.advanced')}
        {advancedOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
      </button>

      {advancedOpen && (
        <div className="space-y-5 rounded-lg border border-border p-4">
          <div className="space-y-2">
            <Label htmlFor="rvc-f0-method">{t('voiceChanger.params.f0Method.label')}</Label>
            <Select
              value={f0Method}
              onValueChange={(value) => onF0Method(value as RvcF0Method)}
              disabled={disabled}
            >
              <SelectTrigger id="rvc-f0-method">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="rmvpe">{t('voiceChanger.params.f0Method.rmvpe')}</SelectItem>
                <SelectItem value="crepe">{t('voiceChanger.params.f0Method.crepe')}</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">{t('voiceChanger.params.f0Method.hint')}</p>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="rvc-index-rate">{t('voiceChanger.params.indexRate.label')}</Label>
              <span className="text-sm tabular-nums text-muted-foreground">
                {indexRate.toFixed(2)}
              </span>
            </div>
            <Slider
              id="rvc-index-rate"
              value={[indexRate]}
              onValueChange={([value]) => onIndexRate(value)}
              min={0}
              max={1}
              step={0.01}
              disabled={disabled || !hasIndex}
              aria-label={t('voiceChanger.params.indexRate.label')}
            />
            <p className="text-xs text-muted-foreground">
              {hasIndex
                ? t('voiceChanger.params.indexRate.hint')
                : t('voiceChanger.params.indexRate.noIndex')}
            </p>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="rvc-rms-mix">{t('voiceChanger.params.rmsMixRate.label')}</Label>
              <span className="text-sm tabular-nums text-muted-foreground">
                {rmsMixRate.toFixed(2)}
              </span>
            </div>
            <Slider
              id="rvc-rms-mix"
              value={[rmsMixRate]}
              onValueChange={([value]) => onRmsMixRate(value)}
              min={0}
              max={1}
              step={0.01}
              disabled={disabled}
              aria-label={t('voiceChanger.params.rmsMixRate.label')}
            />
            <p className="text-xs text-muted-foreground">
              {t('voiceChanger.params.rmsMixRate.hint')}
            </p>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="rvc-protect">{t('voiceChanger.params.protect.label')}</Label>
              <span className="text-sm tabular-nums text-muted-foreground">
                {protect.toFixed(2)}
              </span>
            </div>
            <Slider
              id="rvc-protect"
              value={[protect]}
              onValueChange={([value]) => onProtect(value)}
              min={0}
              max={1}
              step={0.01}
              disabled={disabled}
              aria-label={t('voiceChanger.params.protect.label')}
            />
            <p className="text-xs text-muted-foreground">{t('voiceChanger.params.protect.hint')}</p>
          </div>
        </div>
      )}
    </div>
  );
}
