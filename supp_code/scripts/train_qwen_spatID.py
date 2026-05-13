#!/usr/bin/env python
"""Spatial-ID-loss baseline (Kang et al., ICLR 2026), adapted to 3D scenes.

The original Kang et al. loss adds an auxiliary term that maximizes the
cosine similarity between (a) the projection of an object-token activation
onto a 2D-grid direction basis V_ell = [v_ell, h_ell] in R^{d x 2}, and
(b) a precomputed cross-image-mean ``ground-truth spatial ID'' at the
object's grid position. Adapted to our 3D-continuous setting:

  * V_ell in R^{d x 3} is fitted once before training by orthogonalizing
    the linear-regression coefficients of object-token activations on
    their 3D scene coordinates (held-out subset of train.jsonl).
  * For each per-scene object, the ``ground-truth spatial ID'' is
        Delta_i = V x_i      (= the object's true 3D position written
                              into the d-dim spatial subspace)
  * The model's ``extracted spatial ID'' is
        Delta_hat_i = V V^T h_i
  * Loss:
        L_spatID = 1 - (1/n) sum_i cos(Delta_hat_i, Delta_i)
  * Total: LM_cross_entropy + lambda_sid * L_spatID  (single scalar weight).

Hyperparameter axis matches v2: layer set, lora rank 16, q/k/v/o, bf16,
500 steps, batch 2; we sweep lambda_sid in {0.3, 1, 3} to mirror the
Dirichlet sweep. n=4 seeds.

Imports utilities from train_qwen_dirichlet.py.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from train_qwen_dirichlet import (  # noqa: E402
    JsonlDataset, LayerHook, build_chat_prompt,
    find_obj_positions, make_labels,
)

logger = logging.getLogger("train_spatID")


@dataclass
class Args:
    train_jsonl: Path
    val_jsonl: Path
    output_dir: Path
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    layers: str = "17"  # comma-separated; per-layer mean L_spatID
    lambda_sid: float = 1.0
    steps: int = 500
    batch_size: int = 2
    learning_rate: float = 1e-4
    lora_rank: int = 16
    grad_clip: float = 1.0
    log_every: int = 100
    n_basis: int = 80  # number of scenes to use when fitting V
    seed: int = 0
    lora_targets: str = "q_proj,k_proj,v_proj,o_proj"


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--train-jsonl", type=Path, required=True)
    p.add_argument("--val-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--layers", default="17",
                   help="Comma-separated layer indices (per-layer mean of L_spatID).")
    p.add_argument("--lambda-sid", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--n-basis", type=int, default=80,
                   help="Number of scenes used to fit V before training.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj")
    return Args(**vars(p.parse_args()))


def fit_spatial_basis(
    model, processor, hooks_dict, examples, device, n_max: int,
) -> dict[int, torch.Tensor]:
    """Fit V_ell in R^{d x 3} per layer by orthogonalizing the regression
    coefficients of object-token activations on 3D coords.

    Returns: dict {layer_idx: V_ell (d, 3)} on `device`.
    """
    model.eval()
    H_per_layer: dict[int, list] = {l: [] for l in hooks_dict}
    X_list: list = []
    seen_scenes: dict = {}
    for ex in examples:
        sid = ex.get("scene_id")
        if sid in seen_scenes:
            continue
        seen_scenes[sid] = ex
        if len(seen_scenes) >= n_max:
            break
    with torch.no_grad():
        for ex in seen_scenes.values():
            try:
                user_text, _ = build_chat_prompt(ex)
                image = Image.open(ex["image_path"]).convert("RGB")
                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_text},
                    ]},
                    {"role": "assistant", "content": [{"type": "text", "text": ""}]},
                ]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False)
                inputs = processor(text=[text], images=[image], return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                _ = model(**inputs)
                positions, mask = find_obj_positions(
                    inputs["input_ids"][0], processor.tokenizer, ex["object_names"])
                valid_pos = [p for p, m in zip(positions, mask) if m]
                valid_X = [c for c, m in zip(ex["object_coords"], mask) if m]
                if len(valid_pos) < 2:
                    continue
                for l, hook in hooks_dict.items():
                    H_obj = hook.captured[0, valid_pos].float().cpu().numpy()
                    H_per_layer[l].append(H_obj)
                X_list.append(np.array(valid_X, dtype=np.float32))
            except Exception:
                continue
    if not X_list:
        raise RuntimeError("Could not collect any activations to fit spatial basis.")
    X_all = np.concatenate(X_list, axis=0)  # (N, 3)
    out = {}
    for l in hooks_dict:
        H_all = np.concatenate(H_per_layer[l], axis=0)  # (N, d)
        # Mean-center
        H_c = H_all - H_all.mean(axis=0, keepdims=True)
        X_c = X_all - X_all.mean(axis=0, keepdims=True)
        # Linear regression: solve W in argmin || H_c W - X_c ||^2 -> W = (H_c^+ X_c)
        # Then orthogonalize W's columns to get V (d, 3) with orthonormal columns.
        # W: (d, 3) where W[:, k] is the direction in d-space whose dot product
        # with h best predicts coordinate k.
        W, *_ = np.linalg.lstsq(H_c, X_c, rcond=None)  # (d, 3)
        # Orthonormal basis spanning W's columns
        V, _ = np.linalg.qr(W)  # V: (d, 3) with orthonormal cols
        out[l] = torch.from_numpy(V).to(device).to(torch.float32)
        logger.info("L%d: spatial basis V shape=%s, ||V^TV - I||=%.2e",
                    l, tuple(V.shape),
                    float(np.linalg.norm(V.T @ V - np.eye(3))))
    return out


def spatID_loss_one(H_obj: torch.Tensor, X_obj: torch.Tensor,
                     V: torch.Tensor) -> torch.Tensor:
    """Compute Kang-et-al spatial-ID cosine loss on one scene.

    H_obj: (n_obj, d)  per-object residual activations at this layer
    X_obj: (n_obj, 3)  ground-truth 3D coords
    V:     (d, 3)      orthonormal spatial basis (frozen during training)
    Returns scalar = 1 - mean_i cos(V V^T h_i, V x_i).
    """
    # Mean-center coords (the basis was fit on mean-centered coords)
    X_centered = X_obj - X_obj.mean(dim=0, keepdim=True)
    # Ground-truth spatial ID per object: V @ x_i  (d,)
    Delta_gt = X_centered @ V.T  # (n, d)
    # Extracted spatial ID per object: V V^T h_i  (d,)
    H_proj = H_obj @ V  # (n, 3)
    Delta_hat = H_proj @ V.T  # (n, d)
    # Cosine similarity per object
    cos = F.cosine_similarity(Delta_hat, Delta_gt, dim=-1)  # (n,)
    return 1.0 - cos.mean()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(args.output_dir / "train.log")],
    )
    torch.manual_seed(args.seed)
    import random; random.seed(args.seed)
    np.random.seed(args.seed)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    device = torch.device("cuda:0")

    layer_indices = [int(x) for x in args.layers.split(",") if x.strip()]
    logger.info("Loading model: %s ; layers=%s ; lambda_sid=%g",
                args.model_id, layer_indices, args.lambda_sid)
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import LoraConfig, get_peft_model

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16).to(device)

    target_modules = [t.strip() for t in args.lora_targets.split(",")]
    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank * 2,
        target_modules=target_modules,
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    base = model.base_model.model
    hooks: dict[int, LayerHook] = {}
    for layer_idx in layer_indices:
        layer_module = base.model.language_model.layers[layer_idx]
        hooks[layer_idx] = LayerHook(layer_module)

    train_ds = JsonlDataset(args.train_jsonl)
    val_examples = JsonlDataset(args.val_jsonl).examples
    logger.info("Train: %d  Val: %d", len(train_ds), len(val_examples))

    # Fit the spatial basis V_ell once (frozen for all training).
    logger.info("Fitting spatial-ID basis V on %d held-out val scenes...",
                args.n_basis)
    V_per_layer = fit_spatial_basis(
        model, processor, hooks, val_examples, device, n_max=args.n_basis)
    # Save to disk for traceability
    np.savez(
        args.output_dir / "spatID_basis.npz",
        **{f"V_L{l}": V.detach().cpu().numpy() for l, V in V_per_layer.items()},
    )
    model.train()

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                         collate_fn=lambda b: b)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=0.0,
    )

    history = []
    t0 = time.time()
    step = 0
    train_iter = iter(loader)
    while step < args.steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(loader)
            batch = next(train_iter)

        optim.zero_grad()
        lm_losses, sid_losses = [], []
        for ex in batch:
            try:
                user_text, answer = build_chat_prompt(ex)
                image = Image.open(ex["image_path"]).convert("RGB")
                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_text},
                    ]},
                    {"role": "assistant", "content": [{"type": "text", "text": answer}]},
                ]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False)
                inputs = processor(text=[text], images=[image], return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = make_labels(inputs["input_ids"][0], answer, processor).unsqueeze(0)

                out = model(**inputs, labels=labels)
                lm_loss = out.loss

                positions, mask = find_obj_positions(
                    inputs["input_ids"][0], processor.tokenizer, ex["object_names"])
                valid_pos = [p for p, m in zip(positions, mask) if m]
                valid_X = [c for c, m in zip(ex["object_coords"], mask) if m]

                if len(valid_pos) >= 2 and args.lambda_sid > 0:
                    sid_terms = []
                    for layer_idx, hook in hooks.items():
                        H_all = hook.captured  # (1, T, d)
                        H_obj = H_all[0, valid_pos].float()
                        X_obj = torch.tensor(
                            valid_X, device=device, dtype=torch.float32)
                        V = V_per_layer[layer_idx]
                        sid_terms.append(spatID_loss_one(H_obj, X_obj, V))
                    sid_loss = sum(sid_terms) / len(sid_terms)
                else:
                    sid_loss = torch.zeros((), device=device)

                loss = lm_loss + args.lambda_sid * sid_loss
                loss.backward()
                lm_losses.append(float(lm_loss.detach()))
                sid_losses.append(
                    float(sid_loss.detach()) if torch.is_tensor(sid_loss) else float(sid_loss))
            except Exception as exc:
                logger.warning("step %d: skipping ex (%s)", step, exc)
                continue

        if not lm_losses:
            step += 1
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()

        if step % args.log_every == 0:
            mean_lm = float(np.mean(lm_losses))
            mean_sid = float(np.mean(sid_losses))
            elapsed = time.time() - t0
            logger.info(
                "step %3d/%d  lm=%.4f  sid=%.4f  total=%.4f  (%.1fs/%d)",
                step, args.steps, mean_lm, mean_sid,
                mean_lm + args.lambda_sid * mean_sid, elapsed, step + 1,
            )
            history.append({
                "step": step,
                "lm_loss": mean_lm, "sid_loss": mean_sid, "phase": "train",
            })
        step += 1

    model.save_pretrained(args.output_dir / "lora")
    json.dump({"args": vars(args), "history": history,
                "wall_time_s": time.time() - t0},
               open(args.output_dir / "history.json", "w"),
               indent=2, default=str)
    logger.info("Saved to %s", args.output_dir)
    for h in hooks.values():
        h.close()


if __name__ == "__main__":
    main()
