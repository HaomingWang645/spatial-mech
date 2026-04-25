#!/usr/bin/env python
"""Build a JSONL training set for Dirichlet-loss VLM finetuning.

For each Free6DoF scene, emit one training example per pre-generated
QA pair, with the per-question 3D coordinates of the *involved* objects
attached so the Dirichlet loss can be computed at training time.

Output schema (one JSON object per line):

    {
        "scene_id": "s_001ded82b1_t0",
        "image_path": "data/tier_c_free6dof/s_001ded82b1_t0/frames/000.png",
        "question": "Which is closer to the cyan cylinder: ...",
        "answer": "the yellow cube",
        "involves": [2, 0, 1],
        "object_names": ["cyan cylinder", "yellow cube", "magenta cylinder"],
        "object_coords": [[..3..], [..3..], [..3..]]
    }

Usage
-----
    python scripts/build_dirichlet_train_data.py \\
        --scenes-root data/tier_c_free6dof \\
        --train-out data/dirichlet_train/train.jsonl \\
        --val-out data/dirichlet_train/val.jsonl \\
        --val-frac 0.1
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import random
from pathlib import Path


def object_canonical_name(obj: dict) -> str:
    """Return e.g. 'yellow cube' from {color: 'yellow', shape: 'cube', ...}."""
    return f"{obj['color']} {obj['shape']}"


def build_examples(scene_dir: Path) -> list[dict]:
    scene = json.loads((scene_dir / "scene.json").read_text())
    objects = scene["objects"]
    # Drop duplicates: keep at most one object per (color, shape) so the
    # name->position mapping in the prompt is unambiguous.
    seen = {}
    unique_objs = []
    for o in objects:
        key = (o["color"], o["shape"])
        if key in seen:
            continue
        seen[key] = True
        unique_objs.append(o)
    if len(unique_objs) < 4:
        return []  # Dirichlet needs >= 4 pairs to be meaningful

    all_names = [object_canonical_name(o) for o in unique_objs]
    all_coords = [o["centroid"] for o in unique_objs]
    name_by_id = {o["object_id"]: object_canonical_name(o) for o in objects}

    image_path = str(scene_dir / scene["frames"][0]["image_path"])

    examples = []
    for qa in scene.get("qa", []):
        involves = qa.get("involves", [])
        if not involves:
            continue
        # Skip QA pairs that mention objects not in the unique list (rare).
        try:
            involves_names = [name_by_id[i] for i in involves]
        except KeyError:
            continue
        if not all(n in all_names for n in involves_names):
            continue
        examples.append({
            "scene_id": scene["scene_id"],
            "image_path": image_path,
            "question": qa["question"],
            "answer": qa["answer"],
            "kind": qa.get("kind"),
            # All objects in the scene: used by both the prompt format and
            # the Dirichlet loss.
            "object_names": all_names,
            "object_coords": all_coords,
        })
    return examples


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--scenes-root", type=Path, required=True)
    p.add_argument("--train-out", type=Path, required=True)
    p.add_argument("--val-out", type=Path, required=True)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-scenes", type=int, default=None)
    args = p.parse_args()

    scene_dirs = sorted(d for d in args.scenes_root.iterdir() if d.is_dir())
    if args.max_scenes:
        scene_dirs = scene_dirs[: args.max_scenes]
    print(f"Processing {len(scene_dirs)} scene directories")

    examples = []
    for sd in scene_dirs:
        try:
            examples.extend(build_examples(sd))
        except Exception as exc:  # noqa: BLE001
            print(f"  skipped {sd.name}: {exc}")
    print(f"Built {len(examples)} examples")

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    n_val = max(1, int(len(examples) * args.val_frac))
    val = examples[:n_val]
    train = examples[n_val:]

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.val_out.parent.mkdir(parents=True, exist_ok=True)
    with args.train_out.open("w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")
    with args.val_out.open("w") as f:
        for ex in val:
            f.write(json.dumps(ex) + "\n")

    # Sanity stats
    kinds = {}
    for ex in train:
        kinds[ex.get("kind") or "?"] = kinds.get(ex.get("kind") or "?", 0) + 1
    print(f"Train: {len(train)}  Val: {len(val)}")
    print(f"Train question kinds: {kinds}")


if __name__ == "__main__":
    main()
