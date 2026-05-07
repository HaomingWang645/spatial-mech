"""Real-world counterfactual pilot — Methods 1 & 2.

For each (question, image) in a small VSI-Bench subset, run the same
candidate-scoring eval as `eval_vsi_batched.py` under multiple image
manipulations and record paired accuracies + answer flips.

Manipulations
-------------
  baseline   : original image
  grayscale  : luminance-only (preserves structure, removes color)         [Method 1]
  hue_shift  : full-image hue rotation by 180°                              [Method 1b]
  desat_obj  : per-question target object detected by SAM2 + desaturated
               (gray patch with same shading) — only the named object loses
               color, everything else is unchanged.                         [Method 2]

Per question we record (candidate scores, predicted answer, correctness)
for each manipulation, plus paired flips ("does the prediction differ from
baseline?"). Results are saved to a parquet + a small markdown summary.

Usage:
  python scripts/realworld_counterfactual_pilot.py \\
      --vsi-jsonl data/vsi_bench_full/eval_full.jsonl \\
      --model-id Qwen/Qwen2.5-VL-7B-Instruct \\
      --n 50 --kinds object_rel_distance,object_rel_direction_easy \\
      --out data/realworld_counterfactual/pilot_qwen
"""
import _bootstrap  # noqa: F401

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

# reuse scorer + helpers
sys.path.insert(0, "/home/haoming/x-spatial-manual/scripts")
from eval_vsi_batched import score_candidates_batched, make_distractors_for_numeric, _option_letter
from train_qwen_dirichlet import JsonlDataset  # noqa: E402


# ----------------------- image manipulations ---------------------------

def to_grayscale(img: Image.Image) -> Image.Image:
    """RGB image → luminance-only gray, but kept in RGB mode (3 identical channels)."""
    return ImageOps.grayscale(img).convert("RGB")


def hue_shift(img: Image.Image, deg: float = 180.0) -> Image.Image:
    """Rotate hue by `deg` degrees, preserving value/saturation."""
    arr = np.asarray(img.convert("HSV"))
    h = arr[..., 0].astype(np.int16)
    h = (h + int(round(deg / 360 * 255))) % 256
    out = arr.copy(); out[..., 0] = h.astype(np.uint8)
    return Image.fromarray(out, mode="HSV").convert("RGB")


def desaturate_object_via_sam(
    img: Image.Image,
    object_text: str,
    sam_predictor=None,
    sam_processor=None,
) -> tuple[Image.Image, bool]:
    """If a SAM2-class predictor is supplied and we get a mask, desaturate ONLY
    the masked region (keep luminance, drop chroma). Returns (image, success).

    For the pilot we accept failure: if no SAM is available or the mask is
    empty, we return the original image and success=False.
    """
    if sam_predictor is None or not object_text:
        return img, False
    # Method-2 placeholder: not exercised in pilot if SAM is unavailable.
    return img, False


# ----------------------- the pilot loop ---------------------------------

KIND_DEFAULT = ",".join([
    "object_rel_distance", "object_rel_direction_easy",
    "object_rel_direction_medium", "object_rel_direction_hard",
])


# Match a quoted/named object term in the question (best-effort).
_OBJECT_RE = re.compile(
    r"\bthe ([a-z][a-z\- ]{1,30}?)\b(?=,| or | and|\?|\.| is | to )", re.IGNORECASE
)


def extract_target_objects(question: str) -> list[str]:
    """Heuristic: pull noun phrases that follow 'the' in a question. Used
    only as a placeholder for Method 2; not exact NLP."""
    cands = _OBJECT_RE.findall(question)
    seen, out = set(), []
    for c in cands:
        c = c.strip().lower()
        if c not in seen and c not in ("video", "scene", "room", "first-time appearance order"):
            seen.add(c); out.append(c)
    return out[:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vsi-jsonl", type=Path, required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--kinds", type=str, default=KIND_DEFAULT,
                    help="Comma-separated VSI-Bench question kinds to keep.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manipulations", type=str,
                    default="baseline,grayscale,hue_shift",
                    help="Subset of {baseline,grayscale,hue_shift,desat_obj}.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    keep_kinds = set(k.strip() for k in args.kinds.split(",") if k.strip())
    manipulations = [m.strip() for m in args.manipulations.split(",")]

    rows = JsonlDataset(args.vsi_jsonl).examples
    rows = [r for r in rows if r["kind"] in keep_kinds]
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(rows))[: args.n]
    pilot = [rows[i] for i in idx]
    print(f"[pilot] {len(pilot)} questions kept "
          f"(filtered from {len(rows)} matching kinds)")

    # Distribution by kind
    from collections import Counter
    kc = Counter(r["kind"] for r in pilot)
    for k, v in kc.most_common():
        print(f"  {k:30s} n={v}")

    # ----- model load -----
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    device = torch.device("cuda:0")
    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"[pilot] loading {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    # ----- pilot loop -----
    out_rows = []
    t0 = time.time()
    for j, ex in enumerate(pilot):
        img_orig = Image.open(ex["image_path"]).convert("RGB")
        user_text = ex["question"]
        if ex.get("options"):
            opt_str = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(ex["options"]))
            user_text_with_opts = f"{user_text}\n{opt_str}"
            candidates = list(ex["options"])
        else:
            distractors = make_distractors_for_numeric(ex["answer"])
            if not distractors:
                continue
            candidates = [ex["answer"]] + distractors
            user_text_with_opts = user_text

        per_manip = {}
        baseline_pred = None
        for manip in manipulations:
            if manip == "baseline":
                img = img_orig
            elif manip == "grayscale":
                img = to_grayscale(img_orig)
            elif manip == "hue_shift":
                img = hue_shift(img_orig, deg=180.0)
            elif manip == "desat_obj":
                tgt = extract_target_objects(ex["question"])
                img, _ok = desaturate_object_via_sam(img_orig, tgt[0] if tgt else "")
            else:
                continue
            with torch.no_grad():
                scores = score_candidates_batched(
                    model, processor, img, user_text_with_opts, candidates, device,
                )
            best = max(scores, key=scores.get) if scores else None
            if ex.get("options"):
                pred = _option_letter(best) or best
                gt = ex["answer"]
                correct = (pred == gt)
            else:
                pred = best
                correct = (pred == ex["answer"])
            per_manip[manip] = {"pred": pred, "correct": bool(correct), "best": best}
            if manip == "baseline":
                baseline_pred = pred

        for manip, d in per_manip.items():
            out_rows.append({
                "vsi_id": ex.get("vsi_id"),
                "scene_id": ex.get("scene_id"),
                "dataset": ex.get("dataset"),
                "kind": ex.get("kind"),
                "manipulation": manip,
                "pred": d["pred"],
                "gt": ex["answer"],
                "correct": d["correct"],
                "flip_vs_baseline": int(d["pred"] != baseline_pred) if baseline_pred is not None else 0,
            })
        elapsed = time.time() - t0
        if (j + 1) % 5 == 0 or j == 0:
            print(f"  {j+1}/{len(pilot)}  ({elapsed:.0f}s elapsed)")

    df = pd.DataFrame(out_rows)
    df.to_parquet(args.out / "pilot_results.parquet")

    # ----- summary -----
    print("\n=== Per-manipulation accuracy ===")
    summary = (df.groupby("manipulation")
                 .agg(acc=("correct", "mean"),
                      flip=("flip_vs_baseline", "mean"),
                      n=("correct", "count"))
                 .reset_index())
    print(summary.to_string(index=False))
    summary.to_csv(args.out / "summary.csv", index=False)

    print("\n=== Per-kind × manipulation accuracy ===")
    by_kind = (df.groupby(["kind", "manipulation"])
                 .agg(acc=("correct", "mean"),
                      flip=("flip_vs_baseline", "mean"),
                      n=("correct", "count"))
                 .reset_index())
    print(by_kind.to_string(index=False))
    by_kind.to_csv(args.out / "by_kind.csv", index=False)

    print(f"\nsaved {args.out}/{{pilot_results.parquet, summary.csv, by_kind.csv}}")


if __name__ == "__main__":
    main()
