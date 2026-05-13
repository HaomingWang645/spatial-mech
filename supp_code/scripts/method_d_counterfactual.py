"""Method D — counterfactual color/position swap on synthetic scenes.

For each base scene with ≥4 objects:
- Original (image as-is, original questions)
- Color-swap variant (image with colors permuted; same questions, same positions)
- Position-swap variant (image with positions permuted; same questions; SAME colors)

The model is then asked the original question. We measure how often the answer
*flips* under each variant.

Hypothesis (color-driven head):
- Color-swap: answer changes a lot (model re-locates objects by color, gets wrong position)
- Position-swap: answer follows the new positions (i.e., changes when geometry changes,
  per the question's *intended* spatial relation)

Hypothesis (position-driven head):
- Color-swap: answer should NOT change (positions unchanged → spatial answer unchanged)
- Position-swap: answer reflects the new geometry

This script writes new scene-folders with re-rendered frame 0 only, and a fresh
JSONL with one QA per scene-variant pair.
"""
import _bootstrap  # noqa
import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).parent / "../scripts"))

from spatial_subspace.scene import Scene, Object3D, Frame, Camera
from spatial_subspace.render.tier_c import _draw_frame
from spatial_subspace.utils import load_yaml, ensure_dir


def load_scene(scene_dir):
    sj = json.loads((scene_dir / "scene.json").read_text())
    objs = []
    for o in sj["objects"]:
        objs.append(Object3D(**{k: v for k, v in o.items() if k in
                                  ("object_id", "shape", "color", "size", "centroid",
                                   "bbox_min", "bbox_max", "extras")}))
    frames = []
    for fr in sj["frames"]:
        cam = Camera(intrinsics=fr["camera"]["intrinsics"],
                     extrinsics=fr["camera"]["extrinsics"],
                     kind=fr["camera"].get("kind", "perspective"))
        frames.append(Frame(frame_id=fr["frame_id"], image_path=fr["image_path"],
                              mask_path=fr["mask_path"], camera=cam))
    s = Scene(scene_id=sj["scene_id"], tier=sj["tier"], objects=objs,
              frames=frames, qa=list(sj.get("qa", [])), extras=dict(sj.get("extras", {})))
    return s


def color_swap(scene: Scene, rng: random.Random) -> Scene:
    objs = list(scene.objects)
    perm = list(range(len(objs)))
    rng.shuffle(perm)
    while any(p == i for i, p in enumerate(perm)):
        rng.shuffle(perm)
    new_objs = []
    for i, o in enumerate(objs):
        new = Object3D(object_id=o.object_id, shape=o.shape, color=objs[perm[i]].color,
                       size=o.size, centroid=o.centroid, bbox_min=o.bbox_min,
                       bbox_max=o.bbox_max, extras=o.extras)
        new_objs.append(new)
    return Scene(scene_id=scene.scene_id+"_colorswap", tier=scene.tier,
                 objects=new_objs, frames=scene.frames, qa=list(scene.qa),
                 extras={**scene.extras, "counterfactual": "color_swap"})


def position_swap(scene: Scene, rng: random.Random) -> Scene:
    objs = list(scene.objects)
    perm = list(range(len(objs)))
    rng.shuffle(perm)
    while any(p == i for i, p in enumerate(perm)):
        rng.shuffle(perm)
    new_objs = []
    for i, o in enumerate(objs):
        src = objs[perm[i]]
        new = Object3D(object_id=o.object_id, shape=o.shape, color=o.color, size=o.size,
                       centroid=src.centroid, bbox_min=src.bbox_min, bbox_max=src.bbox_max,
                       extras=o.extras)
        new_objs.append(new)
    return Scene(scene_id=scene.scene_id+"_posswap", tier=scene.tier,
                 objects=new_objs, frames=scene.frames, qa=list(scene.qa),
                 extras={**scene.extras, "counterfactual": "position_swap"})


def render_one_frame(scene: Scene, cfg: dict, frame_idx: int = 0):
    """Render a single frame using the original camera pose."""
    fr = scene.frames[frame_idx]
    eye = np.array(fr.camera.extrinsics)[:3, 3]   # not exact, but cfg-based
    # The renderer's _draw_frame needs eye, target, roll. The original frame's
    # extrinsics are R|t with t=eye. We can re-derive eye/target by inverting.
    E = np.array(fr.camera.extrinsics)
    R = E[:3, :3]; t = E[:3, 3]
    # In the canonical look-at convention: extrinsics map world→camera, so
    # eye_world = -R^T t and "forward" = R^T @ [0,0,-1]^T (if -Z is forward)
    eye_w = -R.T @ t
    fwd_world = R.T @ np.array([0, 0, 1])  # convention varies; tier_c uses +Z forward (look_at returns inverse)
    target_w = eye_w + fwd_world
    img, mask, K, E_new = _draw_frame(scene, cfg, eye_w.tolist(), target_w.tolist(), roll=0.0)
    return img, mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes-dir", type=Path, default=Path("data/tier_c_free6dof"))
    p.add_argument("--out-dir", type=Path, default=Path("data/counterfactuals"))
    p.add_argument("--n-base", type=int, default=30, help="Number of base scenes")
    p.add_argument("--config", type=Path, default=Path("configs/tier_c.yaml"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    rng = random.Random(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scene_ids = sorted([p.name for p in args.scenes_dir.iterdir() if p.is_dir()])
    rng.shuffle(scene_ids)
    base_ids = scene_ids[:args.n_base]
    print(f"Building {len(base_ids)} counterfactual scene triples")

    qa_records = []
    for sid in base_ids:
        try:
            scene = load_scene(args.scenes_dir / sid)
        except Exception as e:
            print(f"  skip {sid}: {e}")
            continue
        if len(scene.objects) < 4:
            continue
        # Color-swap variant
        cs = color_swap(scene, rng)
        # Position-swap variant
        ps = position_swap(scene, rng)
        for variant in [scene, cs, ps]:
            try:
                img, mask = render_one_frame(variant, cfg, frame_idx=0)
                d = args.out_dir / variant.scene_id
                ensure_dir(d / "frames"); ensure_dir(d / "masks")
                img.save(d / "frames" / "000.png")
                mask.save(d / "masks" / "000.png")
                # Write a slim scene.json
                slim = {
                    "scene_id": variant.scene_id,
                    "objects": [
                        {"object_id": o.object_id, "shape": o.shape, "color": o.color,
                         "centroid": o.centroid}
                        for o in variant.objects
                    ],
                    "extras": dict(variant.extras),
                }
                (d / "scene.json").write_text(json.dumps(slim, indent=2))
                # Build QA records
                for qa in variant.qa:
                    qa_records.append({
                        "scene_id": variant.scene_id,
                        "base_scene_id": sid.replace("_t0","").replace("_t1","").replace("_t2","").replace("_t3",""),
                        "variant": variant.extras.get("counterfactual", "original"),
                        "image_path": str((d / "frames" / "000.png").resolve()),
                        "question": qa["question"],
                        "answer": qa["answer"],
                        "kind": qa.get("kind", "relative_position"),
                        "involves": qa.get("involves", []),
                    })
            except Exception as e:
                print(f"  render fail {variant.scene_id}: {e}")

    out_jsonl = args.out_dir / "qa.jsonl"
    with out_jsonl.open("w") as f:
        for r in qa_records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(qa_records)} QA records to {out_jsonl}")
    print(f"Scenes saved under {args.out_dir}")


if __name__ == "__main__":
    main()
