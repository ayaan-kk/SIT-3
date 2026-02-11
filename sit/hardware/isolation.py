"""Process isolation and CPU affinity for hardware experiments.

Uses os.sched_setaffinity when available. Logs all placement
decisions and system state for auditability.
"""

import os
import platform
from typing import Any, Dict, List, Optional

from sit.core.logging import get_logger

logger = get_logger("hardware.isolation")


def get_system_info() -> Dict[str, Any]:
    """Collect system information for auditability.

    Returns:
        Dict with platform, CPU count, OS details.
    """
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count_logical": os.cpu_count() or 0,
        "pid": os.getpid(),
    }

    # Physical cores (if psutil available)
    try:
        import psutil
        info["cpu_count_physical"] = psutil.cpu_count(logical=False) or 0
        info["cpu_freq_mhz"] = psutil.cpu_freq().current if psutil.cpu_freq() else 0
        mem = psutil.virtual_memory()
        info["memory_total_mb"] = mem.total // (1024 * 1024)
        info["memory_available_mb"] = mem.available // (1024 * 1024)
    except ImportError:
        info["cpu_count_physical"] = "unavailable"
        info["cpu_freq_mhz"] = "unavailable"
        info["memory_total_mb"] = "unavailable"

    return info


def try_set_affinity(pid: int, cores: List[int]) -> Dict[str, Any]:
    """Attempt to set CPU affinity for a process.

    Args:
        pid: Process ID.
        cores: List of CPU core indices.

    Returns:
        Dict with affinity status and details.
    """
    result = {
        "pid": pid,
        "requested_cores": cores,
        "affinity_set": False,
        "actual_affinity": "default",
        "method": "none",
    }

    try:
        os.sched_setaffinity(pid, set(cores))
        actual = list(os.sched_getaffinity(pid))
        result["affinity_set"] = True
        result["actual_affinity"] = sorted(actual)
        result["method"] = "os.sched_setaffinity"
        logger.info("Set affinity for pid %d to cores %s", pid, sorted(actual))
    except (AttributeError, OSError, PermissionError) as e:
        result["error"] = str(e)
        result["method"] = "unavailable"
        logger.info("CPU affinity not available: %s", e)

    return result


def get_current_affinity() -> Dict[str, Any]:
    """Get current process CPU affinity.

    Returns:
        Dict with affinity information.
    """
    try:
        cores = sorted(os.sched_getaffinity(0))
        return {"affinity": cores, "available": True}
    except (AttributeError, OSError):
        return {"affinity": "default", "available": False}


def get_background_load() -> Dict[str, Any]:
    """Snapshot of background system load.

    Returns:
        Dict with load average, CPU usage, etc.
    """
    snapshot = {
        "load_avg_1m": -1.0,
        "load_avg_5m": -1.0,
        "load_avg_15m": -1.0,
        "cpu_percent": -1.0,
    }

    try:
        load = os.getloadavg()
        snapshot["load_avg_1m"] = load[0]
        snapshot["load_avg_5m"] = load[1]
        snapshot["load_avg_15m"] = load[2]
    except (OSError, AttributeError):
        pass

    try:
        import psutil
        snapshot["cpu_percent"] = psutil.cpu_percent(interval=0.1)
    except ImportError:
        pass

    # /proc/stat fallback
    try:
        with open("/proc/stat", "r") as f:
            line = f.readline()
            parts = line.split()
            if parts[0] == "cpu":
                snapshot["proc_stat_user"] = int(parts[1])
                snapshot["proc_stat_system"] = int(parts[3])
                snapshot["proc_stat_idle"] = int(parts[4])
    except (FileNotFoundError, PermissionError, IndexError):
        pass

    return snapshot


def log_isolation_context() -> Dict[str, Any]:
    """Log full isolation context for auditability.

    Returns:
        Dict with system info, affinity, background load, scheduling policy.
    """
    context = {
        "system": get_system_info(),
        "affinity": get_current_affinity(),
        "background_load": get_background_load(),
        "scheduling_policy": _get_scheduling_policy(),
    }

    logger.info("Isolation context: cores=%s, load_1m=%.2f",
                context["affinity"].get("affinity", "unknown"),
                context["background_load"].get("load_avg_1m", -1))

    return context


def _get_scheduling_policy() -> str:
    """Get current scheduling policy (best-effort)."""
    try:
        policy = os.sched_getscheduler(0)
        policies = {0: "SCHED_OTHER", 1: "SCHED_FIFO", 2: "SCHED_RR"}
        return policies.get(policy, f"unknown({policy})")
    except (AttributeError, OSError):
        return "unavailable"
