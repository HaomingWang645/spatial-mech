"""Method 3 v2 — depth-aware position swap (real-3D pilot).

Replaces the 2D paste proxy with a depth-aware swap:
  1. Estimate metric depth with Apple DepthPro on the RGB frame.
  2. Use Grounding-DINO + SAM to mask two named objects A, B.
  3. Compute their 2D centroids and median depths z_A, z_B.
  4. Translate each object's pixels to the *other*'s 2D centroid, scaling
     in-place by the depth ratio so apparent size matches the new depth
     (closer = larger). Inpaint the vacated pixels with the local
     background median.

This respects 3D perspective: if A was far and we move it to where B
(close) was, A grows in proportion to z_A/z_B. The result is much closer
to a true 3D position swap than the 2D paste proxy.

Caveats:
 - Monocular depth is scale-ambiguous and noisy at object boundaries.
 - We don't recover full 3D geometry, just a depth-correct 2D edit.
 - Shadows / lighting still come from the original positions.

Usage:
  python scripts/realworld_method3_v2.py \\
      --vsi-jsonl data/vsi_bench_full/eval_full.jsonl \\
      --model-id Qwen/Qwen2.5-VL-7B-Instruct \\
      --n 50 --kinds object_rel_distance,object_rel_direction_easy \\
      --out data/realworld_counterfactual/m3v2_qwen --save-debug-images
"""
import _bootstrap  # noqa: F401

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, "/home/haoming/x-spatial-manual/scripts")
from eval_vsi_batched import score_candidates_batched, _option_letter
from realworld_method2_pilot import GroundedSegmenter
from realworld_method2_v2 import parse_targets
from train_qwen_dirichlet import JsonlDataset  # noqa: E402


# ---------------- depth estimation ----------------

class DepthEstimator:
    def __init__(self, device):
        from transformers import AutoModelForDepthEstimation, AutoImageProcessor
        self.device = device
        self.proc = AutoImageProcessor.from_pretrained("apple/DepthPro-hf")
        self.model = AutoModelForDepthEstimation.from_pretrained(
            "apple/DepthPro-hf", torch_dtype=torch.float32).to(device).eval()

    @torch.no_grad()
    def estimate(self, img: Image.Image) -> np.ndarray:
        """Return (H, W) metric depth in meters (or arbitrary units)."""
        inp = self.proc(images=img, return_tensors="pt").to(self.device)
        out = self.model(**inp)
        pp  = self.proc.post_process_depth_estimation(
            out, target_sizes=[(img.size[1], img.size[0])])
        d = pp[0]["predicted_depth"].cpu().numpy()
        return d


# ---------------- depth-aware swap ----------------

def median_inpaint(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
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


def depth_aware_swap(
    img: Image.Image,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    depth: np.ndarray,
) -> Image.Image:
    """Swap two objects with depth-aware perspective scaling."""
    arr = np.asarray(img).copy()
    H, W = arr.shape[:2]

    def _stat(m):
        ys, xs = np.where(m)
        if len(ys) == 0: return None
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        # Use median depth of masked region (robust to outliers)
        z = float(np.median(depth[m]))
        return cx, cy, max(z, 0.05)  # avoid divide-by-zero

    sA = _stat(mask_a); sB = _stat(mask_b)
    if sA is None or sB is None:
        return img
    cAx, cAy, zA = sA
    cBx, cBy, zB = sB

    pixels_a = arr[mask_a].copy()  # (Na, 3)
    pixels_b = arr[mask_b].copy()
    coords_a = np.column_stack(np.where(mask_a)).astype(np.float32)  # (Na, 2) (y, x)
    coords_b = np.column_stack(np.where(mask_b)).astype(np.float32)

    # Inpaint both source regions first
    arr = median_inpaint(arr, mask_a | mask_b)

    def _move(pixels, coords, src_cx, src_cy, src_z, dst_cx, dst_cy, dst_z):
        # Scale around new centroid by src_z / dst_z (closer = larger)
        scale = src_z / dst_z
        new_y = dst_cy + (coords[:, 0] - src_cy) * scale
        new_x = dst_cx + (coords[:, 1] - src_cx) * scale
        ny = np.clip(np.round(new_y).astype(np.int32), 0, H - 1)
        nx = np.clip(np.round(new_x).astype(np.int32), 0, W - 1)
        return ny, nx

    # Move A → B's location (depth zB)
    ny_a, nx_a = _move(pixels_a, coords_a, cAx, cAy, zA, cBx, cBy, zB)
    arr[ny_a, nx_a] = pixels_a
    # Move B → A's location (depth zA)
    ny_b, nx_b = _move(pixels_b, coords_b, cBx, cBy, zB, cAx, cAy, zA)
    arr[ny_b, nx_b] = pixels_b

    return Image.fromarray(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vsi-jsonl", type=Path, required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=50)
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
    print(f"[m3v2] {len(pilot)} questions kept")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    device = torch.device("cuda:0")

    print("[m3v2] loading depth estimator (DepthPro)")
    depth = DepthEstimator(device)
    print("[m3v2] loading grounded segmenter")
    seg = GroundedSegmenter(device)
    print(f"[m3v2] loading {args.model_id}")
    from transformers import AutoModelForImageTextToText, AutoProcessor
    qproc = AutoProcessor.from_pretrained(args.model_id)
    qmodel = AutoModelForImageTextToText.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16).to(device)
    qmodel.eval()

    out_rows = []
    parse_fail = seg_fail = 0
    t0 = time.time()
    for j, ex in enumerate(pilot):
        targets = parse_targets(ex)
        if targets is None:
            parse_fail += 1; continue
        # Two-object pair: for direction we use (queried, facing); for distance we use (queried_ref, top candidate)
        if "candidates" in targets:
            a, b = targets["queried"], targets["candidates"][0]
        else:
            a, b = targets["queried"], targets.get("facing", targets.get("standing"))
        if a is None or b is None or a == b:
            parse_fail += 1; continue

        img_orig = Image.open(ex["image_path"]).convert("RGB")
        det_a = seg.detect_box(img_orig, a)
        det_b = seg.detect_box(img_orig, b)
        if det_a is None or det_b is None:
            seg_fail += 1; continue
        mask_a = seg.mask_from_box(img_orig, det_a[0])
        mask_b = seg.mask_from_box(img_orig, det_b[0])
        if mask_a.sum() < 200 or mask_b.sum() < 200:
            seg_fail += 1; continue

        d_map = depth.estimate(img_orig)
        img_swap = depth_aware_swap(img_orig, mask_a, mask_b, d_map)

        if args.save_debug_images and j < 30:
            dbg = Image.new("RGB", (img_orig.size[0]*2 + 16, img_orig.size[1]), "white")
            dbg.paste(img_orig, (0, 0))
            dbg.paste(img_swap, (img_orig.size[0] + 16, 0))
            d = ImageDraw.Draw(dbg)
            d.text((4, 4), f"{ex['kind']} :: A={a} B={b}", fill=(255, 255, 255))
            dbg.save(args.out / "debug" / f"{ex.get('vsi_id', j)}.png")

        opt_str = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(ex["options"]))
        user_text = f"{ex['question']}\n{opt_str}"
        candidates = list(ex["options"])
        gt = ex["answer"]

        with torch.no_grad():
            base_scores = score_candidates_batched(qmodel, qproc, img_orig, user_text, candidates, device)
            mod_scores  = score_candidates_batched(qmodel, qproc, img_swap, user_text, candidates, device)

        def _pred(scores):
            best = max(scores, key=scores.get) if scores else None
            return _option_letter(best) or best

        base_pred = _pred(base_scores); mod_pred = _pred(mod_scores)
        out_rows.append({
            "vsi_id": ex.get("vsi_id"),
            "scene_id": ex.get("scene_id"),
            "kind": ex.get("kind"),
            "obj_a": a, "obj_b": b,
            "depth_a": float(np.median(d_map[mask_a])),
            "depth_b": float(np.median(d_map[mask_b])),
            "base_pred": base_pred,
            "mod_pred": mod_pred,
            "gt": gt,
            "base_correct": int(base_pred == gt),
            "mod_correct":  int(mod_pred  == gt),
            "flip": int(base_pred != mod_pred),
        })
        if (j + 1) % 10 == 0 or j == 0:
            n_used = len(out_rows)
            base_acc = np.mean([r["base_correct"] for r in out_rows])
            mod_acc  = np.mean([r["mod_correct"]  for r in out_rows])
            flip     = np.mean([r["flip"]         for r in out_rows])
            print(f"  {j+1}/{len(pilot)} ({time.time()-t0:.0f}s) used={n_used} parse={parse_fail} seg={seg_fail}  base={base_acc:.3f} mod={mod_acc:.3f} flip={flip:.3f}")

    df = pd.DataFrame(out_rows)
    df.to_parquet(args.out / "results.parquet")
    if len(df) == 0:
        print("[m3v2] no usable rows"); return
    n = len(df)
    print(f"\n=== Method 3 v2 (depth-aware position swap, n={n}) ===")
    print(f"  parse_fail / seg_fail / used : {parse_fail} / {seg_fail} / {n}")
    print(f"  baseline acc                  : {df.base_correct.mean():.3f}")
    print(f"  modified acc                  : {df.mod_correct.mean():.3f}")
    print(f"  Δ acc                         : {(df.mod_correct.mean() - df.base_correct.mean()):+.3f}")
    print(f"  flip rate                     : {df.flip.mean():.3f}")
    by_kind = (df.groupby("kind")
                 .agg(base=("base_correct", "mean"),
                      mod =("mod_correct",  "mean"),
                      flip=("flip",         "mean"),
                      n   =("flip",         "count"))
                 .reset_index())
    print("\nby kind:"); print(by_kind.to_string(index=False))
    by_kind.to_csv(args.out / "by_kind.csv", index=False)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
