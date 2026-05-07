"""Method 2 pilot — per-object hue manipulation via Grounding-DINO + SAM.

For each (question, image) we (1) extract a target object phrase from the
VSI-Bench question (regex heuristic), (2) detect a box with grounding-dino
on that phrase, (3) get a mask with SAM, (4) hue-shift only the masked
region. We then re-eval the same question on Qwen2.5-VL-7B with the
modified image and measure (a) prediction flip vs. baseline and (b)
accuracy delta.

This isolates per-object color reliance: only the *named* object's color
changes; everything else (its position, shape, the rest of the scene) is
identical.

Usage:
  python scripts/realworld_method2_pilot.py \\
      --vsi-jsonl data/vsi_bench_full/eval_full.jsonl \\
      --model-id Qwen/Qwen2.5-VL-7B-Instruct \\
      --n 20 \\
      --kinds object_rel_distance,object_rel_direction_easy,object_rel_direction_medium \\
      --out data/realworld_counterfactual/pilot_qwen_method2
"""
import _bootstrap  # noqa: F401

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, "/home/haoming/x-spatial-manual/scripts")
from eval_vsi_batched import score_candidates_batched, make_distractors_for_numeric, _option_letter
from train_qwen_dirichlet import JsonlDataset  # noqa: E402


# -------- target-object extraction (heuristic) --------

# We want short noun phrases right after "the".
_OBJ = re.compile(r"\bthe ([a-z][a-z\- ]{1,25}?)\b(?=,| or | and|\?|\.| is | to | from |\)|$)",
                  re.IGNORECASE)
_STOP = {"video", "scene", "room", "image",
         "first-time appearance order", "longest dimension",
         "closest point of each object", "objects",
         "size of this room", "size of the combined space"}


def extract_target_objects(question: str, options: list | None = None) -> list[str]:
    """Pull short 'the X' noun phrases. Then add option strings for MCQs
    that look like object names (e.g. ``chair, stool, stove``)."""
    cands = _OBJ.findall(question)
    out, seen = [], set()
    for c in cands:
        c = c.strip().lower()
        if c in _STOP or c in seen:
            continue
        if any(c.startswith(s + " ") or c.endswith(" " + s) for s in ("multiple",)):
            continue
        seen.add(c); out.append(c)
    # MCQ options that look like single nouns
    if options:
        for o in options:
            o = str(o).strip().lower()
            if 1 <= len(o.split()) <= 3 and o not in seen and o not in _STOP:
                seen.add(o); out.append(o)
    return out[:4]


# -------- HSV / HSL hue rotation --------

def hue_shift_region(img: Image.Image, mask: np.ndarray, deg: float = 180.0) -> Image.Image:
    """Rotate hue by `deg` degrees on the masked region only."""
    arr_hsv = np.asarray(img.convert("HSV")).copy()
    h = arr_hsv[..., 0].astype(np.int16)
    h_shift = (h + int(round(deg / 360 * 255))) % 256
    if mask.shape[:2] != arr_hsv.shape[:2]:
        # resize mask to image size
        from PIL import Image as _PI
        m_img = _PI.fromarray((mask.astype(np.uint8) * 255))
        m_img = m_img.resize((arr_hsv.shape[1], arr_hsv.shape[0]), _PI.NEAREST)
        mask = (np.asarray(m_img) > 127)
    new_h = np.where(mask, h_shift, h).astype(np.uint8)
    arr_hsv[..., 0] = new_h
    return Image.fromarray(arr_hsv, "HSV").convert("RGB")


# -------- grounding + SAM pipeline --------

class GroundedSegmenter:
    """Grounding-DINO box prediction + SAM mask refinement, both via HF."""

    def __init__(self, device):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from transformers import SamModel, SamProcessor
        self.device = device
        self.gd_proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        self.gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-tiny").to(device).eval()
        self.sam_proc = SamProcessor.from_pretrained("facebook/sam-vit-base")
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device).eval()

    @torch.no_grad()
    def detect_box(self, img: Image.Image, phrase: str, conf=0.30):
        """Return (box xyxy, score) or None."""
        text = phrase.lower().rstrip(".") + "."  # GroundingDINO requires period
        inp = self.gd_proc(images=img, text=text, return_tensors="pt").to(self.device)
        out = self.gd_model(**inp)
        result = self.gd_proc.post_process_grounded_object_detection(
            out, inp["input_ids"],
            threshold=conf, text_threshold=0.20,
            target_sizes=[img.size[::-1]],
        )[0]
        if len(result["boxes"]) == 0:
            return None
        # Take the highest-scoring box
        idx = int(torch.argmax(result["scores"]).item())
        box = result["boxes"][idx].detach().cpu().numpy().tolist()
        score = float(result["scores"][idx].item())
        return box, score

    @torch.no_grad()
    def mask_from_box(self, img: Image.Image, box):
        """SAM mask conditioned on a single box."""
        inp = self.sam_proc(img, input_boxes=[[box]], return_tensors="pt").to(self.device)
        out = self.sam_model(**inp, multimask_output=False)
        masks = self.sam_proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu(),
        )
        m = masks[0][0, 0].numpy()  # (H, W) bool
        return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vsi-jsonl", type=Path, required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--kinds", type=str,
                    default="object_rel_distance,object_rel_direction_easy,"
                            "object_rel_direction_medium,object_rel_direction_hard")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-debug-images", action="store_true",
                    help="Write the modified images to <out>/debug/ for visual inspection.")
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
    print(f"[m2] {len(pilot)} questions kept")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    device = torch.device("cuda:0")

    print("[m2] loading grounded segmenter (Grounding-DINO + SAM)")
    seg = GroundedSegmenter(device)

    print(f"[m2] loading {args.model_id}")
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
        if not targets:
            print(f"  [{j}] no target object; skip")
            continue
        # Try each target until segmentation succeeds; prefer the first.
        chosen_target = None
        chosen_box, chosen_mask = None, None
        for t in targets:
            det = seg.detect_box(img_orig, t)
            if det is None:
                continue
            box, score = det
            mask = seg.mask_from_box(img_orig, box)
            if mask.sum() < 200:  # too small / spurious
                continue
            chosen_target, chosen_box, chosen_mask = t, box, mask
            break
        if chosen_mask is None:
            print(f"  [{j}] no valid mask for {targets[:2]}; skip")
            continue

        # Build modified image (full hue rotation 180° on the masked region)
        img_modif = hue_shift_region(img_orig, chosen_mask, deg=180.0)

        # Optional: save side-by-side debug
        if args.save_debug_images:
            from PIL import ImageOps
            dbg = Image.new("RGB", (img_orig.size[0]*2 + 16, img_orig.size[1]), "white")
            dbg.paste(img_orig, (0, 0))
            dbg.paste(img_modif, (img_orig.size[0] + 16, 0))
            d = ImageDraw.Draw(dbg)
            d.text((4, 4), f"{ex['kind']} :: target={chosen_target}", fill=(0,0,0))
            dbg.save(args.out / "debug" / f"{ex.get('vsi_id', j)}.png")

        # Build user_text + candidates
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
            base_scores = score_candidates_batched(qmodel, qproc, img_orig,    user_text, candidates, device)
            mod_scores  = score_candidates_batched(qmodel, qproc, img_modif,   user_text, candidates, device)

        def _pred(scores):
            best = max(scores, key=scores.get) if scores else None
            return _option_letter(best) or best if ex.get("options") else best

        base_pred, mod_pred = _pred(base_scores), _pred(mod_scores)
        out_rows.append({
            "vsi_id": ex.get("vsi_id"),
            "scene_id": ex.get("scene_id"),
            "dataset": ex.get("dataset"),
            "kind": ex.get("kind"),
            "target_object": chosen_target,
            "base_pred": base_pred,
            "mod_pred": mod_pred,
            "gt": gt,
            "base_correct": int(base_pred == gt),
            "mod_correct":  int(mod_pred  == gt),
            "flip": int(base_pred != mod_pred),
            "mask_pixels": int(chosen_mask.sum()),
        })
        if (j + 1) % 5 == 0 or j == 0:
            print(f"  {j+1}/{len(pilot)}  ({time.time()-t0:.0f}s)  "
                  f"target='{chosen_target}'  base={base_pred} mod={mod_pred}  flip={out_rows[-1]['flip']}")

    df = pd.DataFrame(out_rows)
    df.to_parquet(args.out / "pilot_results.parquet")
    n = len(df)
    if n == 0:
        print("[m2] no usable rows; aborting summary")
        return
    print(f"\n=== Method 2 (per-object hue shift) summary on n={n} questions ===")
    print(f"  baseline acc          : {df.base_correct.mean():.3f}")
    print(f"  modified acc          : {df.mod_correct.mean():.3f}")
    print(f"  Δ acc (mod − base)    : {(df.mod_correct.mean() - df.base_correct.mean()):+.3f}")
    print(f"  flip rate             : {df.flip.mean():.3f}")

    print("\nby kind:")
    by_kind = (df.groupby("kind")
                 .agg(base=("base_correct", "mean"),
                      mod =("mod_correct",  "mean"),
                      flip=("flip",         "mean"),
                      n   =("flip",         "count"))
                 .reset_index())
    print(by_kind.to_string(index=False))
    by_kind.to_csv(args.out / "by_kind.csv", index=False)
    df.describe(include="all").to_csv(args.out / "describe.csv")
    print(f"\nsaved {args.out}/{{pilot_results.parquet, by_kind.csv}}")


if __name__ == "__main__":
    main()
