"""Shared pipeline utilities."""

from __future__ import annotations

from pathlib import Path


def resolve_path(p: Path) -> Path:
    """
    Resolve a path to absolute.

    Args:
        p: Input path.

    Returns:
        ``p`` if it is already absolute, otherwise ``Path(p).resolve()``.
    """
    return p if p.is_absolute() else Path(p).resolve()
