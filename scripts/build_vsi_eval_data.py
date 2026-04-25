#!/usr/bin/env python
"""Build VSI-Bench eval JSONL using overlap with local tier_d ARKitScenes.

For each VSI-Bench question whose scene_name has frames in
data/tier_d/{scene_name}/frames/, emit a JSONL row with:
  - image_path: a frame from tier_d (we use frame 000)
  - question: the VSI-Bench question
  - answer: the ground truth string
  - options: the multiple-choice options (None for numeric questions)
  - kind: VSI-Bench's question_type
  - vsi_id: original VSI-Bench question id

The output is consumable by a VSI-aware evaluator.

Usage
-----
    python scripts/build_vsi_eval_data.py \\
        --tier-d-root data/tier_d \\
        --out data/dirichlet_train_v2/val_vsi_arkit.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier-d-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mc-only", action="store_true",
                   help="If set, only include multiple-choice questions")
    args = p.parse_args()

    from datasets import load_dataset
    ds = load_dataset("nyu-visionx/VSI-Bench", split='test')

    local_scenes = set(os.listdir(args.tier_d_root))
    rows = []
    for ex in ds:
        if ex["dataset"] != "arkitscenes":
            continue
        if ex["scene_name"] not in local_scenes:
            continue
        if args.mc_only and not ex["options"]:
            continue
        frame_dir = args.tier_d_root / ex["scene_name"] / "frames"
        if not frame_dir.exists():
            continue
        # Pick first frame (could use middle or sample multiple)
        frames = sorted(frame_dir.glob("*.png"))
        if not frames:
            continue
        rows.append({
            "vsi_id": ex["id"],
            "scene_id": ex["scene_name"],
            "image_path": str(frames[0]),
            "question": ex["question"],
            "answer": ex["ground_truth"],
            "kind": ex["question_type"],
            "options": ex["options"],
            # No 3D coords or object_names — VSI-Bench evaluator
            # ignores those (Dirichlet loss not applied at eval time)
            "object_names": [],
            "object_coords": [],
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(rows)} examples to {args.out}")
    from collections import Counter
    print("By kind:")
    for k, n in Counter(r["kind"] for r in rows).most_common():
        print(f"  {k}: {n}")
    print("Options-bearing (multiple choice):",
          sum(1 for r in rows if r["options"]))


if __name__ == "__main__":
    main()
