#!/usr/bin/env python
"""Extended Dirichlet training with three new variations:

1. Multi-layer Dirichlet: --layers 13,17,21 applies the loss at all listed
   layers (mean of energies).
2. Online residualization: --online-residualize-every K refreshes the
   nuisance basis W every K training steps from current activations.
3. λ schedule: --lambda-schedule warmup_anneal sets λ(t) = target × min(t/W, 1)
   for t < (T-A), then anneals linearly to 0 over the last A steps.

Imports utilities from train_qwen_dirichlet.py to avoid duplication.
Compatible with --residualize-basis flag from v9.
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
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from train_qwen_dirichlet import (  # noqa: E402
    JsonlDataset, LayerHook, build_chat_prompt, evaluate,
    find_obj_positions, make_labels,
)
from dirichlet_loss import dirichlet_ratio  # noqa: E402

logger = logging.getLogger("train_v2")


@dataclass
class Args:
    train_jsonl: Path
    val_jsonl: Path
    output_dir: Path
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    layers: str = "17"  # comma-separated, e.g., "13,17,21"
    lambda_dir: float = 1.0
    tau: float = 2.0
    steps: int = 500
    batch_size: int = 2
    learning_rate: float = 1e-4
    lora_rank: int = 16
    grad_clip: float = 1.0
    log_every: int = 100
    eval_every: int = 250
    n_eval: int = 100
    seed: int = 0
    residualize_basis: Path | None = None
    online_residualize_every: int = 0  # 0 = no online refit
    lambda_schedule: str = "constant"  # constant | warmup | anneal | warmup_anneal
    warmup_steps: int = 50
    anneal_steps: int = 100
    lora_targets: str = "q_proj,k_proj,v_proj,o_proj"  # comma-sep
    lora_layers: str = ""  # comma-sep layer indices, e.g. "17" or "13,17,21"; empty = all layers


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--train-jsonl", type=Path, required=True)
    p.add_argument("--val-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--layers", default="17", help="Comma-separated layer indices")
    p.add_argument("--lambda-dir", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--n-eval", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--residualize-basis", type=Path, default=None)
    p.add_argument("--online-residualize-every", type=int, default=0)
    p.add_argument("--lambda-schedule", default="constant",
                   choices=["constant", "warmup", "anneal", "warmup_anneal"])
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--anneal-steps", type=int, default=100)
    p.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj")
    p.add_argument("--lora-layers", default="",
                   help="Comma-separated layer indices to apply LoRA to (e.g. '17' or '13,17,21'); "
                        "empty means apply to all layers (default).")
    return Args(**vars(p.parse_args()))


def lambda_at_step(target: float, step: int, total: int,
                   schedule: str, warmup: int, anneal: int) -> float:
    """Compute the effective λ at a given step under the given schedule."""
    if schedule == "constant":
        return target
    elif schedule == "warmup":
        return target * min(step / max(warmup, 1), 1.0)
    elif schedule == "anneal":
        anneal_start = total - anneal
        if step < anneal_start:
            return target
        else:
            t_in_anneal = step - anneal_start
            return target * max(1.0 - t_in_anneal / max(anneal, 1), 0.0)
    elif schedule == "warmup_anneal":
        if step < warmup:
            return target * step / max(warmup, 1)
        anneal_start = total - anneal
        if step < anneal_start:
            return target
        t_in_anneal = step - anneal_start
        return target * max(1.0 - t_in_anneal / max(anneal, 1), 0.0)
    else:
        raise ValueError(f"unknown schedule {schedule}")


def collect_acts_for_basis(
    model, processor, hooks_dict, val_examples, device, n_max: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model on val_examples and collect (H, colors, shapes) at primary layer.

    Used by online_residualize: gather current activations and refit W.
    Uses the LAST hook in hooks_dict (highest layer) as the reference.
    """
    model.eval()
    H_list, color_list, shape_list = [], [], []
    primary_hook = list(hooks_dict.values())[-1]  # highest layer
    seen_scenes = {}
    for ex in val_examples:
        sid = ex.get("scene_id")
        if sid in seen_scenes: continue
        seen_scenes[sid] = ex
        if len(seen_scenes) >= n_max: break
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
                H_all = primary_hook.captured  # (1, T, d)
                positions, mask = find_obj_positions(
                    inputs["input_ids"][0], processor.tokenizer, ex["object_names"])
                for i, (pos, valid) in enumerate(zip(positions, mask)):
                    if not valid: continue
                    h = H_all[0, pos].float().cpu().numpy()
                    name = ex["object_names"][i]
                    parts = name.split(maxsplit=1)
                    if len(parts) == 2:
                        H_list.append(h)
                        color_list.append(parts[0])
                        shape_list.append(parts[1])
            except Exception:
                continue
    if not H_list:
        return None, None, None
    H = np.stack(H_list).astype(np.float32)
    return H, np.array(color_list), np.array(shape_list)


def fit_basis(H: np.ndarray, colors: np.ndarray, shapes: np.ndarray) -> np.ndarray:
    """Fit color+shape probes and return orthonormal basis W ∈ R^(d, k)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def fit_one(labels):
        scaler = StandardScaler()
        X = scaler.fit_transform(H)
        clf = LogisticRegression(max_iter=2000, C=0.1, n_jobs=-1)
        clf.fit(X, labels)
        W_h = clf.coef_ / scaler.scale_
        norms = np.linalg.norm(W_h, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return W_h / norms

    W_color = fit_one(colors)
    W_shape = fit_one(shapes)
    W_stack = np.vstack([W_color, W_shape])
    Q, R = np.linalg.qr(W_stack.T)
    keep = np.abs(np.diag(R)) > 1e-6 * np.abs(np.diag(R)).max()
    return Q[:, keep].astype(np.float32)


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
    logger.info("Loading model: %s ; layers=%s ; schedule=%s ; online_resid=%d",
                args.model_id, layer_indices, args.lambda_schedule,
                args.online_residualize_every)
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import LoraConfig, get_peft_model

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16).to(device)

    target_modules = [t.strip() for t in args.lora_targets.split(",")]
    layers_to_transform = None
    if args.lora_layers.strip():
        layers_to_transform = [int(x.strip()) for x in args.lora_layers.split(",") if x.strip()]
        logger.info("LoRA restricted to layers %s", layers_to_transform)
    lora_kwargs = dict(
        r=args.lora_rank, lora_alpha=args.lora_rank * 2,
        target_modules=target_modules,
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    if layers_to_transform is not None:
        lora_kwargs["layers_to_transform"] = layers_to_transform
    lora_cfg = LoraConfig(**lora_kwargs)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    base_qwen = model.base_model.model
    hooks = {}
    for layer_idx in layer_indices:
        layer_module = base_qwen.model.language_model.layers[layer_idx]
        hooks[layer_idx] = LayerHook(layer_module)
    primary_layer = layer_indices[-1]  # for online basis extraction

    train_ds = JsonlDataset(args.train_jsonl)
    val_examples = JsonlDataset(args.val_jsonl).examples
    logger.info("Train: %d  Val: %d", len(train_ds), len(val_examples))

    # Initial nuisance basis
    nuisance_proj = None
    W_basis = None
    if args.residualize_basis is not None:
        Wnpz = np.load(args.residualize_basis)
        W_basis = torch.from_numpy(Wnpz["W"]).to(device)
        d_in = W_basis.shape[0]
        nuisance_proj = (torch.eye(d_in, device=device) - W_basis @ W_basis.T).to(torch.float32)
        logger.info("Loaded initial nuisance basis W: shape=%s", tuple(W_basis.shape))

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                         collate_fn=lambda b: b)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=0.0,
    )

    history = []
    t0 = time.time()
    step = 0
    model.train()
    train_iter = iter(loader)
    while step < args.steps:
        # Online residualization refresh
        if (args.online_residualize_every > 0 and step > 0 and
                step % args.online_residualize_every == 0):
            logger.info("Online residualize at step %d", step)
            try:
                H, colors, shapes = collect_acts_for_basis(
                    model, processor, hooks, val_examples, device, n_max=40)
                if H is not None and len(set(colors)) >= 2 and len(set(shapes)) >= 2:
                    W_new = fit_basis(H, colors, shapes)
                    W_basis = torch.from_numpy(W_new).to(device)
                    nuisance_proj = (torch.eye(W_basis.shape[0], device=device)
                                     - W_basis @ W_basis.T).to(torch.float32)
                    logger.info("  refit W: shape=%s", tuple(W_basis.shape))
                model.train()
            except Exception as e:
                logger.warning("Online refit failed: %s", e)

        # λ at current step
        lam_t = lambda_at_step(args.lambda_dir, step, args.steps,
                                args.lambda_schedule, args.warmup_steps,
                                args.anneal_steps)

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(loader)
            batch = next(train_iter)

        optim.zero_grad()
        lm_losses, dir_losses = [], []
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

                if len(valid_pos) >= 4 and lam_t > 0:
                    # multi-layer mean of energies
                    dir_terms = []
                    for layer_idx, hook in hooks.items():
                        H_all = hook.captured  # (1, T, d)
                        H_obj = H_all[0, valid_pos].float()
                        if nuisance_proj is not None:
                            H_obj = H_obj @ nuisance_proj
                        X_obj = torch.tensor(valid_X, device=device, dtype=torch.float32)
                        dir_terms.append(dirichlet_ratio(H_obj, X_obj, tau=args.tau))
                    dir_loss = sum(dir_terms) / len(dir_terms)
                else:
                    dir_loss = torch.zeros((), device=device)

                loss = lm_loss + lam_t * dir_loss
                loss.backward()
                lm_losses.append(float(lm_loss.detach()))
                dir_losses.append(float(dir_loss.detach()) if torch.is_tensor(dir_loss) else float(dir_loss))
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
            mean_dir = float(np.mean(dir_losses))
            elapsed = time.time() - t0
            logger.info(
                "step %3d/%d  λ_t=%.3f  lm=%.4f  dir=%.4f  total=%.4f  (%.1fs/%d)",
                step, args.steps, lam_t, mean_lm, mean_dir,
                mean_lm + lam_t * mean_dir, elapsed, step + 1,
            )
            history.append({"step": step, "lambda_t": lam_t,
                             "lm_loss": mean_lm, "dir_loss": mean_dir, "phase": "train"})
        step += 1

    # Save
    model.save_pretrained(args.output_dir / "lora")
    json.dump({"args": vars(args), "history": history,
                "wall_time_s": time.time() - t0},
               open(args.output_dir / "history.json", "w"),
               indent=2, default=str)
    logger.info("Saved to %s", args.output_dir)
    for h in hooks.values(): h.close()


if __name__ == "__main__":
    main()
