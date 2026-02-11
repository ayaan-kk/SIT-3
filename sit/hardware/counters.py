"""Performance counter collection with graceful degradation.

Attempts to use `perf stat` for hardware counters (LLC misses,
cache references, CPU cycles). Falls back gracefully when
perf is unavailable.
"""

import json
import re
import subprocess
from typing import Any, Dict, List, Optional

from sit.core.logging import get_logger

logger = get_logger("hardware.counters")

# Events to collect via perf stat
DEFAULT_PERF_EVENTS = [
    "cache-misses",
    "cache-references",
    "cycles",
    "instructions",
    "LLC-load-misses",
    "LLC-store-misses",
]


def check_perf_available() -> Dict[str, Any]:
    """Check whether perf stat is available.

    Returns:
        Dict with availability status and details.
    """
    try:
        result = subprocess.run(
            ["perf", "stat", "--help"],
            capture_output=True, text=True, timeout=5,
        )
        return {
            "available": result.returncode == 0,
            "method": "perf stat",
            "version": _get_perf_version(),
        }
    except (FileNotFoundError, subprocess.SubprocessError):
        return {
            "available": False,
            "method": "unavailable",
            "reason": "perf not found or not executable",
        }


def _get_perf_version() -> str:
    """Get perf version string."""
    try:
        result = subprocess.run(
            ["perf", "version"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def collect_perf_counters(
    command: List[str],
    events: Optional[List[str]] = None,
    timeout_s: float = 30.0,
) -> Dict[str, Any]:
    """Run a command under perf stat and collect hardware counters.

    Args:
        command: Command to profile (e.g., ["python", "-c", "..."]).
        events: List of perf event names. Uses defaults if None.
        timeout_s: Maximum runtime in seconds.

    Returns:
        Dict with counter values or unavailability status.
    """
    if events is None:
        events = DEFAULT_PERF_EVENTS

    perf_status = check_perf_available()
    if not perf_status["available"]:
        return {
            "counter_status": "unavailable",
            "reason": perf_status.get("reason", "perf not available"),
            "counters": {},
        }

    event_str = ",".join(events)
    perf_cmd = ["perf", "stat", "-e", event_str, "--"] + command

    try:
        result = subprocess.run(
            perf_cmd,
            capture_output=True, text=True,
            timeout=timeout_s,
        )
        counters = _parse_perf_output(result.stderr)
        return {
            "counter_status": "collected",
            "counters": counters,
            "return_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "counter_status": "timeout",
            "reason": f"perf stat timed out after {timeout_s}s",
            "counters": {},
        }
    except (PermissionError, OSError) as e:
        return {
            "counter_status": "permission_denied",
            "reason": str(e),
            "counters": {},
        }


def _parse_perf_output(stderr: str) -> Dict[str, float]:
    """Parse perf stat stderr output into a dict of event -> count.

    Args:
        stderr: Raw perf stat stderr output.

    Returns:
        Dict mapping event name to count value.
    """
    counters = {}
    for line in stderr.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("Performance"):
            continue
        # Typical format: "  1,234,567      cache-misses"
        # or:             "  1234567      cache-misses:u   # 45.2% ..."
        match = re.match(r"^\s*([\d,]+)\s+(\S+)", line)
        if match:
            count_str = match.group(1).replace(",", "")
            event_name = match.group(2).rstrip(":")
            try:
                counters[event_name] = float(count_str)
            except ValueError:
                pass

    return counters


def collect_proc_counters() -> Dict[str, Any]:
    """Collect system counters from /proc as fallback.

    Returns:
        Dict with /proc-based metrics.
    """
    counters = {"counter_status": "proc_fallback"}

    # /proc/stat - CPU info
    try:
        with open("/proc/stat", "r") as f:
            for line in f:
                if line.startswith("cpu "):
                    parts = line.split()
                    counters["cpu_user"] = int(parts[1])
                    counters["cpu_nice"] = int(parts[2])
                    counters["cpu_system"] = int(parts[3])
                    counters["cpu_idle"] = int(parts[4])
                    break
    except (FileNotFoundError, PermissionError, IndexError):
        counters["proc_stat"] = "unavailable"

    # /proc/meminfo
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    counters["mem_total_kb"] = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    counters["mem_available_kb"] = int(line.split()[1])
                elif line.startswith("Cached:"):
                    counters["mem_cached_kb"] = int(line.split()[1])
    except (FileNotFoundError, PermissionError, IndexError):
        counters["proc_meminfo"] = "unavailable"

    return counters


def correlate_counters_with_interference(
    counter_data: List[Dict[str, Any]],
    irbs_deltas: List[float],
    spectator_ids: List[str],
) -> Dict[str, Any]:
    """Correlate hardware counters with IRBS interference deltas.

    Args:
        counter_data: List of counter dicts per spectator.
        irbs_deltas: List of IRBS delta values per spectator.
        spectator_ids: Spectator identifiers.

    Returns:
        Dict with correlation results.
    """
    if not counter_data or not irbs_deltas:
        return {"correlation_status": "no_data"}

    # Find common counter events
    all_events = set()
    for cd in counter_data:
        counters = cd.get("counters", {})
        all_events.update(counters.keys())

    correlations = {}
    for event in all_events:
        values = []
        deltas = []
        for cd, delta in zip(counter_data, irbs_deltas):
            val = cd.get("counters", {}).get(event)
            if val is not None:
                values.append(val)
                deltas.append(delta)

        if len(values) >= 3:
            from scipy import stats as sp_stats
            rho, p_val = sp_stats.spearmanr(values, deltas)
            correlations[event] = {
                "spearman_rho": float(rho) if not (rho != rho) else 0.0,
                "p_value": float(p_val) if not (p_val != p_val) else 1.0,
                "n": len(values),
            }

    return {
        "correlation_status": "computed" if correlations else "insufficient_data",
        "correlations": correlations,
    }
