"""Dataset adapter for rendered scenes."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from .scene import Scene


def iter_scene_dirs(root: str | Path, tier: str | None = None) -> Iterator[Path]:
    root = Path(root)
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not (d / "scene.json").exists():
            continue
        if tier is not None:
            if Scene.load(d).tier != tier:
                continue
        yield d


def load_frame(scene_dir: Path, frame_idx: int = 0) -> tuple[np.ndarray, np.ndarray]:
    scene = Scene.load(scene_dir)
    f = scene.frames[frame_idx]
    img = np.array(Image.open(scene_dir / f.image_path).convert("RGB"))
    mask = np.array(Image.open(scene_dir / f.mask_path))
    return img, mask
