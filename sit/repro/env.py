"""Environment snapshot capture for reproducibility.

Records Python version, OS, CPU model, library versions,
and sets deterministic environment variables.
"""

import os
import platform
import sys
from typing import Any, Dict


def set_deterministic_env(seed: int) -> None:
    """Set environment variables for deterministic execution.

    Args:
        seed: Master seed for the run.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)


def capture_environment() -> Dict[str, Any]:
    """Capture a snapshot of the current execution environment.

    Returns:
        Dict with python_version, os, platform, cpu, and library versions.
    """
    env: Dict[str, Any] = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "os": platform.system(),
        "os_version": platform.version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _get_cpu_model(),
    }

    # Capture key library versions
    libs = {}
    for lib_name in ["numpy", "pandas", "scipy", "yaml", "click"]:
        try:
            mod = __import__(lib_name)
            libs[lib_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            libs[lib_name] = "not installed"
    env["libraries"] = libs

    return env


def _get_cpu_model() -> str:
    """Get CPU model string, best effort."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (FileNotFoundError, PermissionError):
        pass
    return platform.processor() or "unknown"
