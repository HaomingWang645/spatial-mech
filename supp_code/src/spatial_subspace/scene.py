"""Scene dataclasses and JSON serialization.

Shared schema across all tiers: Tier A has one frame, Tier B/C will add more.
On-disk layout per scene is
    <scene_dir>/
        scene.json
        frames/<frame_id>.png
        masks/<frame_id>.png
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Object3D:
    object_id: int
    shape: str
    color: str
    size: str
    centroid: tuple[float, float, float]
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class Camera:
    intrinsics: list[list[float]]
    extrinsics: list[list[float]]
    kind: str = "orthographic"


@dataclass
class Frame:
    frame_id: int
    image_path: str
    mask_path: str
    camera: Camera


@dataclass
class QAItem:
    question: str
    answer: str
    kind: str
    involves: list[int]


@dataclass
class Scene:
    scene_id: str
    tier: str
    objects: list[Object3D]
    frames: list[Frame]
    qa: list[QAItem] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def save(self, scene_dir: str | Path) -> Path:
        scene_dir = Path(scene_dir)
        scene_dir.mkdir(parents=True, exist_ok=True)
        out = scene_dir / "scene.json"
        out.write_text(json.dumps(asdict(self), indent=2))
        return out

    @classmethod
    def load(cls, scene_dir: str | Path) -> "Scene":
        scene_dir = Path(scene_dir)
        raw = json.loads((scene_dir / "scene.json").read_text())
        return _from_dict(raw)


def _t3(x: Any) -> tuple[float, float, float]:
    return (float(x[0]), float(x[1]), float(x[2]))


def _from_dict(raw: dict[str, Any]) -> Scene:
    objects = [
        Object3D(
            object_id=int(o["object_id"]),
            shape=o["shape"],
            color=o["color"],
            size=o["size"],
            centroid=_t3(o["centroid"]),
            bbox_min=_t3(o["bbox_min"]),
            bbox_max=_t3(o["bbox_max"]),
            extras=o.get("extras", {}),
        )
        for o in raw["objects"]
    ]
    frames = [
        Frame(
            frame_id=int(f["frame_id"]),
            image_path=f["image_path"],
            mask_path=f["mask_path"],
            camera=Camera(
                intrinsics=f["camera"]["intrinsics"],
                extrinsics=f["camera"]["extrinsics"],
                kind=f["camera"].get("kind", "orthographic"),
            ),
        )
        for f in raw["frames"]
    ]
    qa = [
        QAItem(
            question=q["question"],
            answer=q["answer"],
            kind=q["kind"],
            involves=list(q["involves"]),
        )
        for q in raw.get("qa", [])
    ]
    return Scene(
        scene_id=raw["scene_id"],
        tier=raw["tier"],
        objects=objects,
        frames=frames,
        qa=qa,
        extras=raw.get("extras", {}),
    )
