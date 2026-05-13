"""Method 3 pilot — 2D pseudo position-swap (proxy for full 3D GS swap).

This is a *proxy* for Method 3's intended 3D Gaussian-Splatting position
swap. We don't have a GS scene library aligned with VSI-Bench camera poses
in this session, so as a stand-in we do a 2D image-space swap:

  1. Pick two object phrases A, B mentioned in the question.
  2. Detect each with Grounding-DINO; refine mask with SAM.
  3. Swap their pixel locations: paste A's masked pixels at B's mask
     centroid (resized to fit B's bbox) and vice versa. Inpaint the empty
     original locations using nearby background median fill.
  4. Re-evaluate Qwen on the modified image.

This is not a 3D-faithful swap — depth, perspective, and shadows are
broken — but it does flip the *2D positional cues* in the image while
keeping object identities and colors intact. If the model's spatial
answer flips here, the model is reading 2D positions; if it doesn't, the
model is grounding spatial relationships some other way (e.g., language
priors).

Usage:
  python scripts/realworld_method3_pilot.py \\
      --vsi-jsonl data/vsi_bench_full/eval_full.jsonl \\
      --model-id Qwen/Qwen2.5-VL-7B-Instruct \\
      --n 15 --kinds object_rel_distance,object_rel_direction_easy \\
      --out data/realworld_counterfactual/pilot_qwen_method3 \\
      --save-debug-images
"""
import _bootstrap  # noqa: F401

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, "/home/haoming/x-spatial-manual/scripts")
from eval_vsi_batched import score_candidates_batched, make_distractors_for_numeric, _option_letter
from realworld_method2_pilot import GroundedSegmenter, extract_target_objects
from train_qwen_dirichlet import JsonlDataset  # noqa: E402


def median_background_fill(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Cheap inpainting: replace the masked region with the per-channel
    median of the unmasked pixels. Smooths the boundary with a small blur.
    """
    out = arr.copy()
    bg = arr[~mask]
    if len(bg) == 0:
        return out
    med = np.median(bg, axis=0).astype(np.uint8)
    out[mask] = med
    img = Image.fromarray(out).filter(ImageFilter.GaussianBlur(radius=1.5))
    arr2 = np.asarray(img).copy()
    out_final = arr.copy()
    out_final[mask] = arr2[mask]
    return out_final


def swap_two_objects(
    img: Image.Image, mask_a: np.ndarray, mask_b: np.ndarray
) -> Image.Image:
    """Swap pixel locations of two masked objects via centroid translation.

    Cheap, non-photorealistic. We paste each object's masked pixels at the
    other's centroid (same scale), inpaint the source region with a median
    background, and feather the boundaries with a small blur."""
    arr = np.asarray(img).copy()  # (H, W, 3)
    H, W = arr.shape[:2]

    def _centroid(m):
        ys, xs = np.where(m)
        if len(ys) == 0:
            return None
        return int(np.mean(xs)), int(np.mean(ys))

    cA, cB = _centroid(mask_a), _centroid(mask_b)
    if cA is None or cB is None:
        return img

    pixels_a = arr[mask_a].copy()
    pixels_b = arr[mask_b].copy()
    coords_a = np.column_stack(np.where(mask_a))  # (N, 2) (y, x)
    coords_b = np.column_stack(np.where(mask_b))

    # First, fill BOTH source regions with background median to avoid
    # double-painting collisions
    arr = median_background_fill(arr, mask_a | mask_b)

    # Paste A pixels at B's centroid
    dy, dx = cB[1] - cA[1], cB[0] - cA[0]
    new_y_a = np.clip(coords_a[:, 0] + dy, 0, H - 1)
    new_x_a = np.clip(coords_a[:, 1] + dx, 0, W - 1)
    arr[new_y_a, new_x_a] = pixels_a

    # Paste B pixels at A's centroid
    dy, dx = cA[1] - cB[1], cA[0] - cB[0]
    new_y_b = np.clip(coords_b[:, 0] + dy, 0, H - 1)
    new_x_b = np.clip(coords_b[:, 1] + dx, 0, W - 1)
    arr[new_y_b, new_x_b] = pixels_b

    return Image.fromarray(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vsi-jsonl", type=Path, required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--kinds", type=str,
                    default="object_rel_distance,object_rel_direction_easy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-debug-images", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.save_debug_images:
        (args.out / "debug").mkdir(parents=True, exist_ok=True)

    keep_kinds = set(k.strip() for k in args.kinds.split(",") if k.strip())
    rows = JsonlDataset(args.vsi_jsonl).examples
    rows = [r for r in rows if r["kind"] in keep_kinds]
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(rows))[: args.n]
    pilot = [rows[i] for i in idx]
    print(f"[m3] {len(pilot)} questions kept")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    device = torch.device("cuda:0")

    print("[m3] loading grounded segmenter")
    seg = GroundedSegmenter(device)

    print(f"[m3] loading {args.model_id}")
    from transformers import AutoModelForImageTextToText, AutoProcessor
    qproc = AutoProcessor.from_pretrained(args.model_id)
    qmodel = AutoModelForImageTextToText.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16).to(device)
    qmodel.eval()

    out_rows = []
    t0 = time.time()
    for j, ex in enumerate(pilot):
        img_orig = Image.open(ex["image_path"]).convert("RGB")
        targets = extract_target_objects(ex["question"], ex.get("options"))
        if len(targets) < 2:
            print(f"  [{j}] need ≥2 targets; got {targets}; skip")
            continue

        # Try the first two targets that both successfully segment
        chosen = []
        for t in targets:
            det = seg.detect_box(img_orig, t)
            if det is None: continue
            box, _ = det
            mask = seg.mask_from_box(img_orig, box)
            if mask.sum() < 200: continue
            chosen.append((t, box, mask))
            if len(chosen) == 2: break
        if len(chosen) < 2:
            print(f"  [{j}] couldn't segment two objects; skip")
            continue

        (tA, _, mA), (tB, _, mB) = chosen
        img_swap = swap_two_objects(img_orig, mA, mB)

        if args.save_debug_images:
            dbg = Image.new("RGB", (img_orig.size[0]*2 + 16, img_orig.size[1]), "white")
            dbg.paste(img_orig, (0, 0))
            dbg.paste(img_swap, (img_orig.size[0] + 16, 0))
            d = ImageDraw.Draw(dbg)
            d.text((4, 4), f"{ex['kind']} :: A={tA}  B={tB}", fill=(255,255,255))
            dbg.save(args.out / "debug" / f"{ex.get('vsi_id', j)}.png")

        if ex.get("options"):
            opt_str = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(ex["options"]))
            user_text = f"{ex['question']}\n{opt_str}"
            candidates = list(ex["options"])
            gt = ex["answer"]
        else:
            distractors = make_distractors_for_numeric(ex["answer"])
            if not distractors:
                continue
            candidates = [ex["answer"]] + distractors
            user_text = ex["question"]
            gt = ex["answer"]

        with torch.no_grad():
            base_scores = score_candidates_batched(qmodel, qproc, img_orig, user_text, candidates, device)
            mod_scores  = score_candidates_batched(qmodel, qproc, img_swap, user_text, candidates, device)

        def _pred(scores):
            best = max(scores, key=scores.get) if scores else None
            return _option_letter(best) or best if ex.get("options") else best

        base_pred, mod_pred = _pred(base_scores), _pred(mod_scores)
        out_rows.append({
            "vsi_id": ex.get("vsi_id"),
            "scene_id": ex.get("scene_id"),
            "kind": ex.get("kind"),
            "obj_a": tA, "obj_b": tB,
            "base_pred": base_pred,
            "mod_pred": mod_pred,
            "gt": gt,
            "base_correct": int(base_pred == gt),
            "mod_correct":  int(mod_pred  == gt),
            "flip": int(base_pred != mod_pred),
        })
        if (j + 1) % 3 == 0 or j == 0:
            print(f"  {j+1}/{len(pilot)} ({time.time()-t0:.0f}s) "
                  f"A='{tA}' B='{tB}' base={base_pred} mod={mod_pred} flip={out_rows[-1]['flip']}")

    df = pd.DataFrame(out_rows)
    df.to_parquet(args.out / "pilot_results.parquet")
    if len(df) == 0:
        print("[m3] no usable rows"); return
    print(f"\n=== Method 3 (2D pseudo position-swap) summary on n={len(df)} ===")
    print(f"  baseline acc       : {df.base_correct.mean():.3f}")
    print(f"  modified acc       : {df.mod_correct.mean():.3f}")
    print(f"  Δ acc (mod - base) : {(df.mod_correct.mean() - df.base_correct.mean()):+.3f}")
    print(f"  flip rate          : {df.flip.mean():.3f}")
    by_kind = (df.groupby("kind")
                 .agg(base=("base_correct","mean"),
                      mod =("mod_correct", "mean"),
                      flip=("flip","mean"),
                      n   =("flip","count"))
                 .reset_index())
    print("\nby kind:"); print(by_kind.to_string(index=False))
    by_kind.to_csv(args.out / "by_kind.csv", index=False)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
