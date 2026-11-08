import logging
import subprocess
import sys

import pytest

from backend.services import voxtral_memory as memory


class FakeMLX:
    def __init__(self):
        self.memory_limit = 70 * memory.GIB
        self.cache_limit = 70 * memory.GIB
        self.active = 2 * memory.GIB
        self.clears = 0

    def device_info(self):
        return {"memory_size": 64 * memory.GIB}

    def set_memory_limit(self, limit):
        previous, self.memory_limit = self.memory_limit, limit
        return previous

    def set_cache_limit(self, limit):
        previous, self.cache_limit = self.cache_limit, limit
        return previous

    def clear_cache(self):
        self.clears += 1

    def reset_peak_memory(self):
        pass

    def get_active_memory(self):
        return self.active

    def get_peak_memory(self):
        return self.active

    def get_cache_memory(self):
        return 0


@pytest.fixture
def guarded_memory(monkeypatch):
    monkeypatch.delenv(memory.MEMORY_ENV, raising=False)
    monkeypatch.setattr(memory, "process_footprint", lambda: 3 * memory.GIB)
    monkeypatch.setattr(memory, "_diagnostic_logger", lambda: logging.getLogger("test-voxtral-memory"))
    return FakeMLX()


@pytest.mark.parametrize(("total", "expected"), [(64, 8), (16, 6.4), (8, 3.2)])
def test_budget_reserves_ram_on_smaller_machines(guarded_memory, total, expected):
    assert memory.memory_budget(total * memory.GIB) == int(expected * memory.GIB)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "typo"])
def test_invalid_budget_fails_before_inference(guarded_memory, monkeypatch, value):
    monkeypatch.setenv(memory.MEMORY_ENV, value)
    with pytest.raises(ValueError, match=r"positive finite|could not convert"):
        memory.VoxtralMemoryGuard(guarded_memory, "test")


def test_override_cannot_consume_all_system_ram(guarded_memory, monkeypatch):
    monkeypatch.setenv(memory.MEMORY_ENV, "1000")
    assert memory.memory_budget(16 * memory.GIB) == int(6.4 * memory.GIB)


def test_limits_and_monitor_are_restored_after_success(guarded_memory):
    with memory.VoxtralMemoryGuard(guarded_memory, "test") as guard:
        assert guarded_memory.memory_limit == 8 * memory.GIB
        assert guarded_memory.cache_limit == memory.CACHE_LIMIT
        guard.check()
    assert not guard.thread.is_alive()
    assert guarded_memory.memory_limit == 70 * memory.GIB
    assert guarded_memory.cache_limit == 70 * memory.GIB


def test_limit_is_enforced_and_restored_after_active_memory_spike(guarded_memory):
    def spike(guard):
        guarded_memory.active = 9 * memory.GIB
        guard.check()

    with (
        pytest.raises(memory.VoxtralMemoryError, match="budget exceeded"),
        memory.VoxtralMemoryGuard(guarded_memory, "test") as guard,
    ):
        spike(guard)
    assert not guard.thread.is_alive()
    assert guarded_memory.memory_limit == 70 * memory.GIB


def test_preexisting_pressure_refuses_to_start(guarded_memory, monkeypatch):
    monkeypatch.setattr(memory, "process_footprint", lambda: 9 * memory.GIB)
    guard = memory.VoxtralMemoryGuard(guarded_memory, "test")
    with pytest.raises(memory.VoxtralMemoryError), guard:
        pytest.fail("Inference must not start")
    assert guard.thread is None
    assert guarded_memory.memory_limit == 70 * memory.GIB
    assert guarded_memory.cache_limit == 70 * memory.GIB


def test_transient_process_spike_is_not_forgotten(guarded_memory, monkeypatch):
    guard = memory.VoxtralMemoryGuard(guarded_memory, "test")
    monkeypatch.setattr(memory, "process_footprint", lambda: 9 * memory.GIB)
    guard._sample()
    monkeypatch.setattr(memory, "process_footprint", lambda: 3 * memory.GIB)
    with pytest.raises(memory.VoxtralMemoryError):
        guard.check()


def test_monitor_failure_cancels_cooperatively(guarded_memory, monkeypatch):
    guard = memory.VoxtralMemoryGuard(guarded_memory, "test")

    def unreadable():
        raise OSError("probe failed")

    monkeypatch.setattr(memory, "process_footprint", unreadable)
    guard._monitor()
    assert guard.exceeded.is_set()
    assert guard.monitor_error == "probe failed"


def test_emergency_stop_exits_only_the_child_process(tmp_path):
    # Exercise the real os._exit path without allocating gigabytes or risking
    # the test runner. No MLX import or model download is needed.
    script = """
import logging
import threading
from backend import config
from backend.services import voxtral_memory as m
import sys
config.set_data_dir(sys.argv[1])
g = object.__new__(m.VoxtralMemoryGuard)
g.stopped = threading.Event()
g.exceeded = threading.Event()
g.peak_footprint = 0
g.budget = 8 * m.GIB
g.emergency = 10 * m.GIB
g.operation = 'test native runaway'
g.logger = m._diagnostic_logger()
m.process_footprint = lambda: 11 * m.GIB
g._monitor()
raise AssertionError('Emergency stop did not exit')
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, timeout=10)
    assert result.returncode == 75, result.stderr.decode()
    log = (tmp_path / "logs" / "voxtral-memory.log").read_text()
    assert "emergency memory stop" in log
    assert "11.00 GiB" in log


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS libproc ABI")
def test_native_footprint_probe():
    assert memory.process_footprint() > 0


def test_native_probe_errors_are_not_reported_as_zero(monkeypatch):
    monkeypatch.setattr(memory, "_rusage_function", lambda: lambda *args: -1)
    with pytest.raises(OSError, match="Cannot read"):
        memory.process_footprint()
