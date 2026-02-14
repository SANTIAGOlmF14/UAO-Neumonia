from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd().resolve()
