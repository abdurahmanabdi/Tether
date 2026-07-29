"""Versioned prompt templates on disk, addressed by name + version.

A prompt is a file, not a string in code: `prompts/<name>/<version>.md`.
That makes a prompt change a diff, a diff reviewable, and a version pinnable
from the eval CLI (`--prompt-version v2`).
"""

from __future__ import annotations

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


class PromptNotFound(FileNotFoundError):
    pass


def load_prompt(name: str, version: str) -> str:
    path = PROMPTS_DIR / name / f"{version}.md"
    if not path.is_file():
        available = sorted(
            p.relative_to(PROMPTS_DIR).as_posix() for p in PROMPTS_DIR.glob("*/*.md")
        )
        raise PromptNotFound(
            f"no prompt {name}/{version}; available: {', '.join(available) or 'none'}"
        )
    return path.read_text(encoding="utf-8")


def list_versions(name: str) -> list[str]:
    directory = PROMPTS_DIR / name
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.md"))
