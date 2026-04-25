#!/usr/bin/env python
"""Real Dirichlet-loss finetuning of Qwen2.5-VL-7B on Free6DoF spatial VQA.

Loads the HF model directly (bypassing the inference-only wrapper),
applies LoRA, registers a forward hook at the chosen LLM layer, and
trains with combined `lm_cross_entropy + lambda * dirichlet_ratio`.

Usage
-----
    CUDA_VISIBLE_DEVICES=4 python scripts/train_qwen_dirichlet.py \\
        --train-jsonl data/dirichlet_train/train.jsonl \\
        --val-jsonl data/dirichlet_train/val.jsonl \\
        --output-dir checkpoints/qwen7b_dir_lam0.1 \\
        --layer 17 --lambda-dir 0.1 --tau 2.0 \\
        --steps 200 --batch-size 1 --eval-every 50

The script trains for `--steps` gradient updates; total compute is
linear in this.  `--lambda-dir 0` runs the baseline (LM-only).
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

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from dirichlet_loss import dirichlet_ratio  # noqa: E402

logger = logging.getLogger("train")


# --------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------- #


class JsonlDataset(Dataset):
    def __init__(self, path: Path):
        self.examples = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def build_chat_prompt(ex: dict) -> tuple[str, str]:
    """Build (user-prompt-text, assistant-answer-text) for chat template."""
    obj_list = ", ".join(ex["object_names"])
    user_text = (
        f"The scene contains: {obj_list}.\n"
        f"{ex['question']}"
    )
    return user_text, ex["answer"]


# --------------------------------------------------------------------- #
# Object-token-position extraction
# --------------------------------------------------------------------- #


def find_obj_positions(
    input_ids_row: torch.Tensor,
    tokenizer,
    names: list[str],
) -> tuple[list[int], list[bool]]:
    """For each object name, return the position in `input_ids_row` of
    the LAST token of its first occurrence.  Returns (positions, mask).
    """
    row = input_ids_row.tolist()
    positions, mask = [], []
    for name in names:
        # Try a few tokenizations (with/without leading space)
        candidates = [
            tokenizer.encode(name, add_special_tokens=False),
            tokenizer.encode(" " + name, add_special_tokens=False),
        ]
        found = -1
        for cand in candidates:
            n = len(cand)
            if n == 0:
                continue
            for i in range(len(row) - n + 1):
                if row[i : i + n] == cand:
                    found = i + n - 1
                    break
            if found >= 0:
                break
        positions.append(max(found, 0))
        mask.append(found >= 0)
    return positions, mask


# --------------------------------------------------------------------- #
# Build labels: supervise only the assistant's answer tokens.
# --------------------------------------------------------------------- #


def make_labels(
    input_ids: torch.Tensor,
    answer_text: str,
    processor,
) -> torch.Tensor:
    """Mask all tokens except those that come from the answer.

    Strategy: find the *last* occurrence of the answer's tokenization
    in input_ids and supervise from there to the end.  This is robust
    even when the answer appears once in the prompt's question.
    """
    labels = input_ids.clone()
    labels[:] = -100  # default: ignore
    tok_ids = processor.tokenizer.encode(answer_text, add_special_tokens=False)
    if not tok_ids:
        return labels
    n = len(tok_ids)
    seq = input_ids.tolist()
    last_match = -1
    for i in range(len(seq) - n + 1):
        if seq[i : i + n] == tok_ids:
            last_match = i
    if last_match >= 0:
        labels[last_match : last_match + n] = input_ids[last_match : last_match + n]
    return labels


# --------------------------------------------------------------------- #
# Forward hook
# --------------------------------------------------------------------- #


class LayerHook:
    """Capture residual stream output at one transformer block."""
    def __init__(self, layer_module: nn.Module):
        self.captured: torch.Tensor | None = None
        self._h = layer_module.register_forward_hook(self._fn)

    def _fn(self, _mod, _inp, out):
        self.captured = out[0] if isinstance(out, tuple) else out

    def close(self):
        self._h.remove()


# --------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------- #


@dataclass
class Args:
    train_jsonl: Path
    val_jsonl: Path
    output_dir: Path
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    layer: int = 17
    lambda_dir: float = 0.1
    tau: float = 2.0
    steps: int = 200
    batch_size: int = 1
    learning_rate: float = 1e-4
    lora_rank: int = 8
    grad_clip: float = 1.0
    log_every: int = 5
    eval_every: int = 50
    n_eval: int = 50
    seed: int = 0


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--train-jsonl", type=Path, required=True)
    p.add_argument("--val-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--layer", type=int, default=17)
    p.add_argument("--lambda-dir", type=float, default=0.1)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--n-eval", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    return Args(**vars(p.parse_args()))


def _build_distractor(ex: dict) -> str | None:
    """Build a wrong-answer string for log-prob comparison."""
    answer = ex["answer"].strip().lower()
    kind = ex.get("kind")
    if kind == "relative_position":
        # "yes"/"no" toggle
        if answer == "yes":
            return "no"
        if answer == "no":
            return "yes"
        return None
    if kind == "distance_order":
        # Question lists two candidates: "...the X or the Y?"
        # Answer is one of them; distractor is the other.
        q = ex["question"].lower()
        if " or " not in q:
            return None
        # Crudely split on "the ... or the ..."
        try:
            tail = q.split(":", 1)[1].rstrip("?").strip()  # ": the X or the Y"
            options = [o.strip(" .") for o in tail.split(" or ")]
            options = [o for o in options if o]
            for o in options:
                if o != answer:
                    return o
        except Exception:  # noqa: BLE001
            return None
    return None


def _logprob_of_answer(model, processor, ex, candidate, device) -> float | None:
    """Total log-prob of `candidate` as the assistant turn, given the prompt."""
    image = Image.open(ex["image_path"]).convert("RGB")
    user_text, _ = build_chat_prompt(ex)
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": candidate}]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels = make_labels(inputs["input_ids"][0], candidate, processor).unsqueeze(0)
    n_supervised = int((labels != -100).sum())
    if n_supervised == 0:
        return None
    with torch.no_grad():
        out = model(**inputs, labels=labels)
    # out.loss is mean over supervised tokens; total log-prob is -loss * count
    return float(-out.loss * n_supervised)


@torch.no_grad()
def evaluate(
    model, processor, hook, val_examples, layer_idx, tau, device,
    n_eval, vqa_accuracy: bool = False,
):
    """Compute mean Dirichlet ratio + mean alignment R² + LM loss + (optional) VQA accuracy."""
    model.eval()

    def alignment(H, X):
        Hc = H - H.mean(0); Xc = X - X.mean(0)
        if Hc.shape[0] < 4:
            return float("nan")
        U, S, _ = torch.linalg.svd(Hc.float(), full_matrices=False)
        top3 = U[:, :3] * S[:3]
        try:
            A = torch.linalg.lstsq(top3, Xc.float()).solution
        except RuntimeError:
            return float("nan")
        ss_res = (top3 @ A - Xc.float()).pow(2).sum()
        ss_tot = Xc.float().pow(2).sum() + 1e-8
        return float(1 - ss_res / ss_tot)

    drs, aligns, losses = [], [], []
    vqa_correct, vqa_total = 0, 0

    for ex in val_examples[:n_eval]:
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
            n_supervised = int((labels != -100).sum())
            out = model(**inputs, labels=labels)
            losses.append(float(out.loss))

            H_all = hook.captured  # (1, T, d)
            positions, mask = find_obj_positions(
                inputs["input_ids"][0], processor.tokenizer, ex["object_names"])
            valid_pos = [p for p, m in zip(positions, mask) if m]
            valid_X = [c for c, m in zip(ex["object_coords"], mask) if m]
            if len(valid_pos) >= 4:
                H_obj = H_all[0, valid_pos].float()
                X_obj = torch.tensor(valid_X, device=device, dtype=torch.float32)
                drs.append(float(dirichlet_ratio(H_obj, X_obj, tau=tau)))
                aligns.append(alignment(H_obj, X_obj))

            # VQA accuracy via log-prob comparison vs distractor
            if vqa_accuracy:
                lp_true = float(-out.loss * n_supervised) if n_supervised > 0 else None
                distractor = _build_distractor(ex)
                if lp_true is not None and distractor:
                    lp_dist = _logprob_of_answer(
                        model, processor, ex, distractor, device)
                    if lp_dist is not None:
                        # Length-normalize so longer answers don't auto-win.
                        n_dist = len(processor.tokenizer.encode(
                            distractor, add_special_tokens=False))
                        if n_dist > 0:
                            avg_true = lp_true / max(n_supervised, 1)
                            avg_dist = lp_dist / n_dist
                            if avg_true > avg_dist:
                                vqa_correct += 1
                            vqa_total += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("eval skipped: %s", exc)

    model.train()
    import numpy as np
    return {
        "val_lm_loss": float(np.mean(losses)) if losses else float("nan"),
        "val_dirichlet_ratio": float(np.mean(drs)) if drs else float("nan"),
        "val_alignment_R2": float(np.mean(aligns)) if aligns else float("nan"),
        "val_n": len(losses),
        "vqa_accuracy": (vqa_correct / vqa_total) if vqa_total > 0 else float("nan"),
        "vqa_n": vqa_total,
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.output_dir / "train.log"),
        ],
    )
    torch.manual_seed(args.seed)
    import random; random.seed(args.seed)
    import numpy as np; np.random.seed(args.seed)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    device = torch.device("cuda:0")

    logger.info("Loading model: %s", args.model_id)
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from peft import LoraConfig, get_peft_model

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16,
    ).to(device)
    # NOTE: do NOT enable gradient checkpointing -- it's incompatible with
    # forward-hook-based loss terms.  H100 has plenty of memory for full
    # activations of Qwen2.5-VL-7B at batch_size=1.

    # LoRA on the LLM only
    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Hook at the chosen LLM layer.
    # Path after PEFT wrap: base_model.model is the original Qwen2_5_VLForConditionalGeneration,
    # whose .model.language_model.layers is the LLM stack.
    base_qwen = model.base_model.model  # peeling LoraModel + PeftModel wrappers
    layer_module = base_qwen.model.language_model.layers[args.layer]
    hook = LayerHook(layer_module)

    train_ds = JsonlDataset(args.train_jsonl)
    val_examples = JsonlDataset(args.val_jsonl).examples
    logger.info("Train: %d  Val: %d", len(train_ds), len(val_examples))

    # No collate; we process one example at a time (image preprocessing is variable-size).
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
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(loader)
            batch = next(train_iter)

        # Per-sample loss accumulation
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
                labels = make_labels(
                    inputs["input_ids"][0], answer, processor).unsqueeze(0)

                out = model(**inputs, labels=labels)
                lm_loss = out.loss

                H_all = hook.captured  # (1, T, d)
                positions, mask = find_obj_positions(
                    inputs["input_ids"][0], processor.tokenizer, ex["object_names"])
                valid_pos = [p for p, m in zip(positions, mask) if m]
                valid_X = [c for c, m in zip(ex["object_coords"], mask) if m]

                if len(valid_pos) >= 4 and args.lambda_dir > 0:
                    H_obj = H_all[0, valid_pos].float()
                    X_obj = torch.tensor(
                        valid_X, device=device, dtype=torch.float32)
                    dir_loss = dirichlet_ratio(H_obj, X_obj, tau=args.tau)
                else:
                    dir_loss = torch.zeros((), device=device)

                loss = lm_loss + args.lambda_dir * dir_loss
                loss.backward()
                lm_losses.append(float(lm_loss))
                dir_losses.append(float(dir_loss))
            except Exception as exc:  # noqa: BLE001
                logger.warning("step %d: skipping ex (%s)", step, exc)
                continue

        if not lm_losses:
            logger.warning("step %d: empty batch", step)
            step += 1
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()

        if step % args.log_every == 0:
            import numpy as np
            mean_lm = float(np.mean(lm_losses))
            mean_dir = float(np.mean(dir_losses))
            elapsed = time.time() - t0
            logger.info(
                "step %3d/%d  lm=%.4f  dir=%.4f  total=%.4f  (%.1fs/%d=%.2fs/step)",
                step, args.steps, mean_lm, mean_dir,
                mean_lm + args.lambda_dir * mean_dir, elapsed,
                step + 1, elapsed / (step + 1),
            )
            history.append({
                "step": step, "lm_loss": mean_lm, "dir_loss": mean_dir,
                "phase": "train",
            })

        if (step + 1) % args.eval_every == 0:
            ev = evaluate(
                model, processor, hook, val_examples,
                args.layer, args.tau, device, args.n_eval,
            )
            logger.info(
                "  [eval @ step %d] lm=%.4f  dir=%.4f  R²=%.4f  (n=%d)",
                step, ev["val_lm_loss"], ev["val_dirichlet_ratio"],
                ev["val_alignment_R2"], ev["val_n"],
            )
            history.append({"step": step, "phase": "eval", **ev})

        step += 1

    # Final eval — full val set + VQA accuracy
    ev = evaluate(
        model, processor, hook, val_examples,
        args.layer, args.tau, device, len(val_examples),
        vqa_accuracy=True,
    )
    logger.info(
        "FINAL  lm=%.4f  dir=%.4f  R²=%.4f  vqa_acc=%.4f (n=%d)",
        ev["val_lm_loss"], ev["val_dirichlet_ratio"], ev["val_alignment_R2"],
        ev["vqa_accuracy"], ev["vqa_n"],
    )
    history.append({"step": args.steps, "phase": "final", **ev})

    # Save history + LoRA weights
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
    model.save_pretrained(args.output_dir / "lora")
    logger.info("Saved to %s", args.output_dir)
    hook.close()


if __name__ == "__main__":
    main()
