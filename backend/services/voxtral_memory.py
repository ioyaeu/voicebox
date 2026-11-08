"""Voxtral memory budget, checked without touching MLX from the monitor thread.

MLX 0.31's allocator limit is advisory. Bound codec windows in the backend,
cancel cooperatively at the budget, and exit this server as a last resort if
native inference exceeds the emergency threshold before yielding again.
"""

import ctypes
import logging
import math
import os
import threading
from functools import lru_cache
from logging.handlers import RotatingFileHandler

GIB = 1024**3
CACHE_LIMIT = 256 * 1024**2
MEMORY_ENV = "VOICEBOX_VOXTRAL_MEMORY_GB"


class VoxtralMemoryError(RuntimeError):
    """Inference exceeded its memory budget and must release its model."""


class _RusageInfoV0(ctypes.Structure):
    # Stable macOS libproc ABI from sys/resource.h; footprint includes compressed
    # memory, unlike RSS, which can look small during a severe pressure event.
    _fields_ = [
        ("uuid", ctypes.c_uint8 * 16),
        *[
            (name, ctypes.c_uint64)
            for name in (
                "user_time",
                "system_time",
                "pkg_idle_wkups",
                "interrupt_wkups",
                "pageins",
                "wired_size",
                "resident_size",
                "phys_footprint",
                "proc_start_abstime",
                "proc_exit_abstime",
            )
        ],
    ]


@lru_cache(maxsize=1)
def _rusage_function():
    lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    function = lib.proc_pid_rusage
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    function.restype = ctypes.c_int
    return function


def process_footprint() -> int:
    usage = _RusageInfoV0()
    if _rusage_function()(os.getpid(), 0, ctypes.byref(usage)) != 0:
        raise OSError(ctypes.get_errno(), "Cannot read Voicebox memory footprint")
    return usage.phys_footprint


def memory_budget(total_memory: int) -> int:
    requested = float(os.environ.get(MEMORY_ENV, "8"))
    if not math.isfinite(requested) or requested <= 0:
        raise ValueError(f"{MEMORY_ENV} must be a positive finite number of GiB")
    if total_memory <= 0:
        raise ValueError("Cannot determine system RAM for the Voxtral memory budget")
    return min(int(requested * GIB), int(total_memory * 0.4))


def _diagnostic_logger():
    from ..config import get_data_dir

    logger = logging.getLogger("voicebox.voxtral-memory")
    if not logger.handlers:
        directory = get_data_dir() / "logs"
        directory.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            directory / "voxtral-memory.log", maxBytes=256 * 1024, backupCount=2, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class VoxtralMemoryGuard:
    """One load or inference job; enter/exit on the serial MLX worker only."""

    def __init__(self, mx, operation: str):
        self.mx = mx
        self.operation = operation
        self.budget = memory_budget(int(mx.device_info()["memory_size"]))
        self.emergency = self.budget * 5 // 4
        self.logger = _diagnostic_logger()
        self.stopped = threading.Event()
        self.exceeded = threading.Event()
        self.monitor_error = None
        self.peak_footprint = 0
        self.thread = None
        self.previous_memory = None
        self.previous_cache = None

    def _sample(self):
        footprint = process_footprint()
        self.peak_footprint = max(self.peak_footprint, footprint)
        if footprint >= self.budget:
            self.exceeded.set()
        return footprint

    def _monitor(self):
        while not self.stopped.wait(0.1):
            try:
                footprint = self._sample()
            except Exception as exc:
                self.monitor_error = str(exc)
                self.exceeded.set()
                return
            if footprint >= self.emergency:
                message = (
                    f"Voxtral emergency memory stop: {footprint / GIB:.2f} GiB >= "
                    f"{self.emergency / GIB:.2f} GiB during {self.operation}; "
                    "restart the Voicebox server (exit 75)."
                )
                # Persist before exit; no model unload or Metal calls from here.
                try:
                    self.logger.critical(message)
                finally:
                    os._exit(75)

    def check(self):
        footprint = self._sample()
        active = self.mx.get_active_memory()
        if self.exceeded.is_set() or active >= self.budget:
            raise VoxtralMemoryError(
                f"Voxtral memory budget exceeded ({self.budget / GIB:.2f} GiB; "
                f"process {footprint / GIB:.2f} GiB, MLX active {active / GIB:.2f} GiB). "
                "Generation stopped; the model will be released."
                + (f" Memory monitor failed: {self.monitor_error}" if self.monitor_error else "")
            )

    def __enter__(self):
        try:
            self.previous_memory = self.mx.set_memory_limit(self.budget)
            self.previous_cache = self.mx.set_cache_limit(CACHE_LIMIT)
            self.mx.clear_cache()
            self.mx.reset_peak_memory()
            self.check()
            self.logger.info(
                "%s start: budget=%.2f GiB emergency=%.2f GiB footprint=%.2f GiB",
                self.operation,
                self.budget / GIB,
                self.emergency / GIB,
                self.peak_footprint / GIB,
            )
            self.thread = threading.Thread(target=self._monitor, name="voxtral-memory", daemon=True)
            self.thread.start()
        except BaseException:
            self._restore_limits()
            raise
        return self

    def _restore_limits(self):
        try:
            if self.previous_memory is not None:
                self.mx.set_memory_limit(self.previous_memory)
        finally:
            if self.previous_cache is not None:
                self.mx.set_cache_limit(self.previous_cache)

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.stopped.set()
        if self.thread is not None:
            self.thread.join()
        try:
            self.logger.info(
                "%s %s: footprint_peak=%.2f GiB mlx_peak=%.2f GiB active=%.2f GiB cache=%.2f GiB",
                self.operation,
                "failed" if exc_type else "finished",
                self.peak_footprint / GIB,
                self.mx.get_peak_memory() / GIB,
                self.mx.get_active_memory() / GIB,
                self.mx.get_cache_memory() / GIB,
            )
        finally:
            self._restore_limits()
