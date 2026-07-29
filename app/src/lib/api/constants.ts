// Single source of truth for RVC frontend defaults.
//
// These mirror the backend defaults (`ConvertRequest` in `backend/models.py`
// and `DEFAULT_RVC_BASE_VOICE` in `backend/services/profiles.py`). Keep them in
// sync with the backend — every RVC panel imports from here so the values are
// defined exactly once on the frontend.
import type { RvcConvertParams } from './types';

/** Default base voice for the TTS→RVC chain (`"{engine}:{voice_id}"`). */
export const DEFAULT_RVC_BASE_VOICE = 'kokoro:af_heart';

/** Default RVC conversion knobs — mirrors backend `ConvertRequest` defaults. */
export const RVC_DEFAULTS: Required<RvcConvertParams> = {
  f0_up_key: 0,
  f0_method: 'rmvpe',
  index_rate: 0.75,
  rms_mix_rate: 0.25,
  protect: 0.33,
};
