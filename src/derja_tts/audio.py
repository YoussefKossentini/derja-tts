"""Small audio helpers shared by the data-prep scripts. No heavy imports at module level."""

from __future__ import annotations

import math
import re
from pathlib import Path

Region = tuple[float, float]


def group_speech(
    regions: list[Region],
    min_dur: float = 4.0,
    max_dur: float = 12.0,
    max_gap: float = 0.8,
) -> list[Region]:
    """Merge VAD speech regions into training clips.

    Consecutive regions are joined while the clip stays under `max_dur` and the
    pause between them is under `max_gap` seconds. A clip shorter than `min_dur`
    is dropped. A single region longer than `max_dur` is split into equal parts
    (this cuts mid-speech, so record with natural pauses).
    """
    clips: list[Region] = []
    current: list[float] | None = None

    def flush() -> None:
        if current and current[1] - current[0] >= min_dur:
            clips.append((current[0], current[1]))

    for start, end in sorted(regions):
        if end - start > max_dur:
            flush()
            current = None
            n = math.ceil((end - start) / max_dur)
            step = (end - start) / n
            for k in range(n):
                piece = (start + k * step, start + (k + 1) * step)
                if piece[1] - piece[0] >= min_dur:
                    clips.append(piece)
            continue
        if current is None:
            current = [start, end]
        elif end - current[0] <= max_dur and start - current[1] <= max_gap:
            current[1] = end
        else:
            flush()
            current = [start, end]
    flush()
    return clips


def pad(region: Region, pad_s: float, total: float) -> Region:
    return max(0.0, region[0] - pad_s), min(total, region[1] + pad_s)


def safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_") or "clip"


def write_metadata(path: Path, rows: list[tuple[str, str, str]]) -> None:
    """LJSpeech layout: id|raw text|normalized text (no header). Pipes in text are removed."""
    lines = ["|".join(col.replace("|", " ").replace("\n", " ").strip() for col in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
