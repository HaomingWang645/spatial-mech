"""Method 2 v2 — per-object hue manipulation with template-aware target extraction.

Same intervention as `realworld_method2_pilot.py` but with a much cleaner
target-object parser:

  object_rel_direction_{easy,medium,hard}:
      "If I am standing by the X and facing the Y, is the Z to the ... ?"
      → 3 named objects (X = standing, Y = facing, Z = queried)
      We hue-shift the *queried* object Z (the one the answer is about).

  object_rel_distance:
      "...which of these objects (A, B, C, D) is the closest to the REF?"
      → 4 candidate objects + 1 reference REF
      We hue-shift REF (changing the colour of the named anchor for the
      distance comparison).

Both cases isolate a single named object and rotate its hue 180°. We
re-evaluate the same model on the modified image and record per-question
flip + correctness.

Usage:
  python scripts/realworld_method2_v2.py \\
      --vsi-jsonl data/vsi_bench_full/eval_full.jsonl \\
      --model-id Qwen/Qwen2.5-VL-7B-Instruct \\
      --n 300 --kinds object_rel_distance,object_rel_direction_easy,object_rel_direction_medium,object_rel_direction_hard \\
      --out data/realworld_counterfactual/m2v2_qwen
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
from PIL import Image, ImageDraw

sys.path.insert(0, "/home/haoming/x-spatial-manual/scripts")
from eval_vsi_batched import score_candidates_batched, _option_letter
from realworld_method2_pilot import GroundedSegmenter, hue_shift_region
from train_qwen_dirichlet import JsonlDataset  # noqa: E402


# -------- template-aware target-object parser --------

DIR_RE  = re.compile(
    r"If I am standing by the ([a-z][a-z\- ]+?) and facing the ([a-z][a-z\- ]+?),"
    r" is the ([a-z][a-z\- ]+?) (?:to my|to the)",
    re.IGNORECASE)

DIST_RE = re.compile(
    r"which of these objects \(([^)]+)\) is the closest to the ([a-z][a-z\- ]+?)\?",
    re.IGNORECASE)


def parse_targets(ex: dict) -> dict | None:
    """Returns {"queried": "<obj>"} on success, None on parse failure.

    For direction questions we hue-shift the *queried* object (Z). For
    distance questions we hue-shift the *reference* object (REF). In both
    cases this is the object whose colour identity is most directly tied
    to the answer.
    """
    kind = ex["kind"]
    q = ex["question"]
    if kind.startswith("object_rel_direction"):
        m = DIR_RE.search(q)
        if not m:
            return None
        # X = standing, Y = facing, Z = queried object
        x, y, z = m.group(1).strip().lower(), m.group(2).strip().lower(), m.group(3).strip().lower()
        return {"queried": z, "standing": x, "facing": y}
    if kind == "object_rel_distance":
        m = DIST_RE.search(q)
        if not m:
            return None
        cands = [c.strip().lower() for c in m.group(1).split(",")]
        ref = m.group(2).strip().lower()
        return {"queried": ref, "candidates": cands}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vsi-jsonl", type=Path, required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--kinds", type=str,
                    default="object_rel_distance,object_rel_direction_easy,"
                            "object_rel_direction_medium,object_rel_direction_hard")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-debug-images", action="store_true")
    ap.add_argument("--max-attempts-per-q", type=int, default=2,
                    help="If primary target can't be segmented, fall back to one alt")
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
    print(f"[m2v2] {len(pilot)} questions kept (target = {args.n})")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    device = torch.device("cuda:0")

    print("[m2v2] loading grounded segmenter")
    seg = GroundedSegmenter(device)

    print(f"[m2v2] loading {args.model_id}")
    from transformers import AutoModelForImageTextToText, AutoProcessor
    qproc = AutoProcessor.from_pretrained(args.model_id)
    qmodel = AutoModelForImageTextToText.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16).to(device)
    qmodel.eval()

    out_rows = []
    parse_fail = 0
    seg_fail = 0
    t0 = time.time()
    for j, ex in enumerate(pilot):
        targets = parse_targets(ex)
        if targets is None:
            parse_fail += 1; continue
        primary = targets["queried"]
        # Build a short fallback list of alternative named objects
        alts = []
        if "candidates" in targets:
            alts.extend([c for c in targets["candidates"] if c != primary])
        for k in ("standing", "facing"):
            if k in targets and targets[k] != primary:
                alts.append(targets[k])
        candidates_to_try = [primary] + alts[: args.max_attempts_per_q]

        img_orig = Image.open(ex["image_path"]).convert("RGB")
        chosen, mask = None, None
        for t in candidates_to_try:
            det = seg.detect_box(img_orig, t)
            if det is None: continue
            box, _ = det
            m = seg.mask_from_box(img_orig, box)
            if m.sum() < 200: continue
            chosen, mask = t, m
            break
        if mask is None:
            seg_fail += 1; continue

        img_modif = hue_shift_region(img_orig, mask, deg=180.0)

        if args.save_debug_images and j < 30:
            dbg = Image.new("RGB", (img_orig.size[0]*2 + 16, img_orig.size[1]), "white")
            dbg.paste(img_orig, (0, 0))
            dbg.paste(img_modif, (img_orig.size[0] + 16, 0))
            d = ImageDraw.Draw(dbg)
            d.text((4, 4), f"{ex['kind']} :: target={chosen}", fill=(255, 255, 255))
            dbg.save(args.out / "debug" / f"{ex.get('vsi_id', j)}.png")

        # Build user_text + candidates
        opt_str = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(ex["options"]))
        user_text = f"{ex['question']}\n{opt_str}"
        candidates = list(ex["options"])
        gt = ex["answer"]

        with torch.no_grad():
            base_scores = score_candidates_batched(qmodel, qproc, img_orig, user_text, candidates, device)
            mod_scores  = score_candidates_batched(qmodel, qproc, img_modif, user_text, candidates, device)

        def _pred(scores):
            best = max(scores, key=scores.get) if scores else None
            return _option_letter(best) or best

        base_pred = _pred(base_scores)
        mod_pred  = _pred(mod_scores)
        out_rows.append({
            "vsi_id": ex.get("vsi_id"),
            "scene_id": ex.get("scene_id"),
            "kind": ex.get("kind"),
            "target_object": chosen,
            "target_role": "queried" if chosen == primary else "fallback",
            "base_pred": base_pred,
            "mod_pred": mod_pred,
            "gt": gt,
            "base_correct": int(base_pred == gt),
            "mod_correct":  int(mod_pred  == gt),
            "flip": int(base_pred != mod_pred),
            "mask_pixels": int(mask.sum()),
        })
        if (j + 1) % 25 == 0 or j == 0:
            n_done = len(out_rows)
            base_acc = np.mean([r["base_correct"] for r in out_rows])
            mod_acc  = np.mean([r["mod_correct"]  for r in out_rows])
            flip     = np.mean([r["flip"]         for r in out_rows])
            print(f"  {j+1}/{len(pilot)} ({time.time()-t0:.0f}s) used={n_done} parse_fail={parse_fail} seg_fail={seg_fail} base={base_acc:.3f} mod={mod_acc:.3f} flip={flip:.3f}")

    df = pd.DataFrame(out_rows)
    df.to_parquet(args.out / "results.parquet")
    if len(df) == 0:
        print("[m2v2] no usable rows"); return
    n = len(df)
    print(f"\n=== Method 2 v2 (per-object hue, n={n}) ===")
    print(f"  parse_fail / seg_fail / used : {parse_fail} / {seg_fail} / {n} (total tried = {len(pilot)})")
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
    print(f"\nsaved {args.out}/{{results.parquet, by_kind.csv}}")


if __name__ == "__main__":
    main()
