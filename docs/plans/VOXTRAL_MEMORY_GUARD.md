# Voxtral Memory Guard

## Incident and cause

On September 8, 2026, a running macOS server had a physical-footprint peak
of **85.9 GiB** on a 64 GiB machine, then returned to 3.2 GiB (`vmmap -summary`).
Kernel logs around 12:50-12:51 local time also recorded memory reclamation
under pressure. RSS alone understated the server's memory after compression.
There were no per-generation memory measurements in the previous build, so
the exact offending text cannot be attributed from those logs.

The installed stack is MLX 0.31.2 and mlx-audio 0.4.3. Voxtral's audio codec
materializes full T-by-T attention scores and masks even with a sliding-window
mask. Voicebox previously decoded all audio frames together. Its default
4096-frame generation budget permits approximately 328 seconds of audio for
a single text chunk when end-of-audio prediction fails. This is a credible
mechanism for a transient quadratic memory spike, rather than proof of a
permanent retained-object leak.

## Protection

- The existing text splitter respects Voxtral's 320-character backend ceiling,
  including `speak`, speech sessions, manual generation and RVC base speech.
  Direct backend calls above that ceiling fail before loading.
- Voxtral uses mlx-audio's native `stream=True`, with two-second codec windows.
  The library includes and trims prior context. Voicebox copies and concatenates
  only new samples, exactly once, and does not replay a full final waveform.
  This internal decoding mode does not change the public WAV response contract.
- Audio-token budgets scale with input length, allowing slow speech and pauses,
  with a 60-second absolute ceiling. Reaching the budget fails the generation;
  partial/truncated audio is not published as a successful response.
- Each Voxtral load/generation has a default **8 GiB** process-footprint/MLX-active
  budget and a **256 MiB** allocator cache. A lower budget is selected on smaller
  Macs: at most 40% of physical RAM. `VOICEBOX_VOXTRAL_MEMORY_GB` can adjust the
  target before server startup, still bounded to 40% of RAM. Invalid values fail
  before inference. GiB means 1024 cubed bytes.
- The macOS probe includes compressed memory via `proc_pid_rusage`, not just RSS.
  The worker checks between decoder windows. A 100 ms monitor remembers transient
  over-budget samples and requests cooperative failure at the next checkpoint.
  The generator is closed, its model references released on failure, and cached
  buffers cleared on the MLX worker. The next request can reload the model.
- At **125% of the budget** (10 GiB with the default on a 64 GiB Mac), the monitor
  logs an emergency and exits the backend with code 75. This last resort releases
  native allocations even if inference has not yielded. Restart the Voicebox
  server from the app after this emergency; automatic restart is not implemented.
  The monitor never unloads a model or calls Metal from its own thread.

The allocator's [`set_memory_limit`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_memory_limit.html)
is only a guideline, not a hard OS allocation quota. The monitor is sampled and
can overshoot between samples or while its thread cannot run. Bounded codec
windows remove the known large allocation pattern. These protections apply
during Voxtral work, not to arbitrary other engines or the entire desktop app.
Allocator settings are restored on exit so other engines retain their policy.

## Diagnostics and verification

The rotating `<Voicebox data directory>/logs/voxtral-memory.log` records budget,
operation, character count, voice, sampled footprint peak, MLX peak and remaining
active/cache sizes. Each file is capped at 256 KiB, with two backups. No spoken
text is logged. On macOS the default data directory is
`~/Library/Application Support/sh.voicebox.app`.

Regression tests cover native footprint reads, budget validation, restored
allocator settings, pressure before loading, cooperative failure, iterator
cleanup, recovery on the next request, exact sample concatenation, and emergency
exit in a child process with simulated memory measurements (no large allocation).

Local validation on the incident machine (September 8, 2026): four consecutive
300-character segments, alternating French female/male, produced 15.92-18.96
seconds of finite, non-silent audio each. Sampled physical-footprint peaks were
3.49-3.56 GiB; MLX peak was 3.17 GiB and returned to 2.34 GiB of model weights
with an empty cache after each segment. All 36 codec calls stayed at or below
41 input frames (25 new frames plus up to 16 context frames).

Separate real-model fault tests shortened the token budget and lowered the
memory budget to 3.2 GiB. Both stopped the generation and returned MLX active
and cached allocations to zero. A subsequent normal-budget request reloaded
the model and generated speech successfully. The emergency-exit path was
verified in a child process with simulated measurements, not a real OOM.

The related regression suite passed 66 tests with one skip. These checks used
the Python runtime, not a rebuilt application bundle. Longer daily-use testing
and listening at codec-window boundaries remain necessary; this fix does not
claim to repair intermittent whisper/echo generated by the model.
