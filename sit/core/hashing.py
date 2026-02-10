"""Deterministic hashing for configs and artifacts.

Provides stable, reproducible hashes for:
- Config dicts (key-sorted, float-normalized SHA-256)
- Files on disk (SHA-256 of contents)
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def _normalize_value(v: Any) -> Any:
    """Normalize a value for stable JSON serialization.

    Floats are rounded to 12 significant digits to avoid platform-dependent
    repr differences. Dicts are sorted recursively. Lists are processed
    element-wise.
    """
    if isinstance(v, float):
        # Normalize to 12 decimal places for cross-platform stability
        return round(v, 12)
    elif isinstance(v, dict):
        return {k: _normalize_value(val) for k, val in sorted(v.items())}
    elif isinstance(v, (list, tuple)):
        return [_normalize_value(item) for item in v]
    return v


def hash_config(config_dict: Dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 hash of a config dictionary.

    The dict is recursively sorted by key, floats are normalized to
    stable repr, and the result is serialized to JSON with sorted keys
    and no whitespace variation.

    Args:
        config_dict: Configuration dictionary to hash.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    normalized = _normalize_value(config_dict)
    canonical_json = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    """Compute SHA-256 hash of a file's contents.

    Reads the file in 64KB chunks to handle large files efficiently.

    Args:
        path: Path to the file.

    Returns:
        Hex-encoded SHA-256 digest string.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    h = hashlib.sha256()
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot hash non-existent file: {path}")
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
