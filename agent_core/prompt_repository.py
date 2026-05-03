from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class PromptRepository:
    base_dir: Path
    _cache: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.base_dir = self.base_dir.resolve()

    def read(self, relative_path: str) -> str:
        normalized = relative_path.replace("\\", "/").strip().lstrip("/")
        cached = self._cache.get(normalized)
        if cached is not None:
            return cached

        prompt_path = (self.base_dir / normalized).resolve()
        try:
            prompt_path.relative_to(self.base_dir)
        except ValueError as exc:
            raise ValueError(f"Prompt path escapes prompt repository: {relative_path}") from exc

        if not prompt_path.is_file():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        content = prompt_path.read_text(encoding="utf-8", errors="replace").strip()
        self._cache[normalized] = content
        return content


def load_prompt(base_dir: Path, relative_path: str) -> str:
    return PromptRepository(base_dir).read(relative_path)
