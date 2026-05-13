#!/usr/bin/env python
"""Finetune a VLM with the Dirichlet-ratio regularizer.

Combines standard language-modelling cross-entropy with the
3D-geometry-weighted Dirichlet ratio of :mod:`scripts.dirichlet_loss`,
applied to a chosen layer of the residual stream.

This is a *template*: the dataset adapter (`make_dataset`) and
object-token-position extractor (`extract_object_token_positions`) are
problem-specific and need to be filled in for the user's data layout.
The default implementations target the existing
``data/tier_c_free6dof`` layout used elsewhere in this repository.

Example
-------
    python scripts/train_with_dirichlet.py \\
        --model-config configs/models/qwen25vl_3b.yaml \\
        --layer 17 --lambda-dir 0.1 --tau 1.0 \\
        --train-root data/tier_c_free6dof \\
        --output-dir checkpoints/dirichlet_l17_lam0.1 \\
        --epochs 1 --batch-size 4

Dependencies: torch, transformers >= 4.45, peft (for LoRA), and the
existing ``spatial_subspace`` package for model-specific utilities.

This script intentionally avoids using ``transformers.Trainer``: the
custom loss requires per-step access to the residual stream, which is
cleaner with an explicit training loop.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (sets sys.path)

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dirichlet_loss import DirichletLoss, ResidualStreamHook  # type: ignore

logger = logging.getLogger("train_with_dirichlet")


# --------------------------------------------------------------------- #
# Dataset adapter
# --------------------------------------------------------------------- #


@dataclass
class TrainExample:
    """One training sample."""

    image_path: Path
    question: str
    answer: str
    object_names: list[str] = field(default_factory=list)
    """Canonical names of objects referenced in ``question``, in the
    same order as their 3D coordinates appear in ``object_coords``."""
    object_coords: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, 3)
    )
    """``(n_obj, 3)`` ground-truth 3D world coordinates."""


class Free6DofDirichletDataset(Dataset):
    """Loads (image, question, answer, object_coords) tuples from
    ``data/tier_c_free6dof`` for Dirichlet-regularized training.

    Each scene's JSON file contains the per-object 3D coordinates we
    need for ``object_coords``; the question/answer pairs come from
    the corresponding spatial-VQA file (generated separately).
    """

    def __init__(
        self,
        scenes_root: Path,
        vqa_path: Path,
        max_objects: int = 8,
    ):
        self.scenes_root = Path(scenes_root)
        self.vqa = json.loads(Path(vqa_path).read_text())
        self.max_objects = max_objects

    def __len__(self) -> int:
        return len(self.vqa)

    def __getitem__(self, idx: int) -> TrainExample:
        item = self.vqa[idx]
        scene_id = item["scene_id"]
        scene = json.loads((self.scenes_root / f"{scene_id}.json").read_text())

        # Map object names referenced in the question to their 3D coords
        coords = []
        names = []
        for obj_name in item["objects_in_question"]:
            for obj in scene["objects"]:
                if obj["name"] == obj_name:
                    coords.append(obj["world_position"])
                    names.append(obj_name)
                    break
        if len(coords) == 0:
            # No referenced objects with coords — return a dummy
            coords = [[0.0, 0.0, 0.0]]
            names = ["__none__"]

        return TrainExample(
            image_path=self.scenes_root / item["image_path"],
            question=item["question"],
            answer=item["answer"],
            object_names=names[: self.max_objects],
            object_coords=torch.tensor(
                coords[: self.max_objects], dtype=torch.float32
            ),
        )


# --------------------------------------------------------------------- #
# Object-token-position extractor
# --------------------------------------------------------------------- #


def extract_object_token_positions(
    tokenizer,
    input_ids: torch.Tensor,
    object_names: list[list[str]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find which positions in ``input_ids`` correspond to each object.

    Strategy: for each sample, for each object name, find the first
    occurrence of the object's tokens in the input sequence.  Returns
    a ``(B, n_obj)`` index tensor and a ``(B, n_obj)`` validity mask.

    For tokenizers that split object words into multiple subwords (e.g.
    "rubber duck" → 3 tokens), we use the *first* token's position.
    This is consistent with the extraction convention in
    ``spatial_subspace.extract``.
    """
    batch_size = input_ids.shape[0]
    max_obj = max(len(n) for n in object_names) if object_names else 0
    indices = torch.zeros(batch_size, max_obj, dtype=torch.long)
    mask = torch.zeros(batch_size, max_obj, dtype=torch.bool)

    for b, names in enumerate(object_names):
        for i, name in enumerate(names):
            if name == "__none__":
                continue
            # Tokenize object name (without leading space) and look for
            # its first token in input_ids[b].  This is a heuristic;
            # production code should use offset_mapping to be robust.
            tok_ids = tokenizer.encode(name, add_special_tokens=False)
            if not tok_ids:
                continue
            first = tok_ids[0]
            row = input_ids[b].tolist()
            try:
                pos = row.index(first)
            except ValueError:
                continue
            indices[b, i] = pos
            mask[b, i] = True

    return indices, mask


# --------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------- #


@dataclass
class TrainArgs:
    model_config: Path
    train_root: Path
    train_vqa: Path
    output_dir: Path
    layer: int
    lambda_dir: float = 0.1
    tau: float = 1.0
    epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 1e-4
    lora_rank: int = 16
    grad_clip: float = 1.0
    log_every: int = 10
    save_every_steps: int = 500
    seed: int = 0


def build_model(model_config_path: Path, lora_rank: int) -> tuple[nn.Module, Any]:
    """Load a VLM with LoRA adapters attached.

    This calls into the existing ``spatial_subspace`` model wrappers so
    that the same model that's used for activation extraction can be
    finetuned without architecture mismatches.  The wrapper is
    expected to expose:
       - ``.model``: the underlying nn.Module (the LLM with vision
         tower attached)
       - ``.tokenizer``: the text tokenizer
       - ``.layer_module(idx)``: a method returning the transformer
         block at index ``idx`` (used for the residual hook)
    """
    from spatial_subspace.utils import load_yaml
    from spatial_subspace.models import (
        InternVL3Wrapper,
        LlavaOnevisionWrapper,
        Qwen25VLWrapper,
    )
    from peft import LoraConfig, get_peft_model

    cfg = load_yaml(model_config_path)
    family = cfg["family"]
    wrapper_cls = {
        "qwen25vl": Qwen25VLWrapper,
        "internvl3": InternVL3Wrapper,
        "llava_onevision": LlavaOnevisionWrapper,
    }[family]
    wrapper = wrapper_cls(**cfg["init_kwargs"])

    # Wrap the LLM with LoRA — leave the vision tower frozen
    lora = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    wrapper.model = get_peft_model(wrapper.model, lora)
    wrapper.model.print_trainable_parameters()
    return wrapper.model, wrapper


def collate(batch: list[TrainExample], tokenizer) -> dict[str, Any]:
    """Pack a batch of TrainExamples into model-ready tensors."""
    # NOTE: in practice you also need the image tensors and any
    # vision-side preprocessing; we omit it from this template since
    # it's wrapper-specific.
    texts = [f"Q: {ex.question}\nA: {ex.answer}" for ex in batch]
    enc = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    )
    object_names = [ex.object_names for ex in batch]
    # Pad object_coords to max_obj across batch
    max_obj = max(ex.object_coords.shape[0] for ex in batch)
    coords = torch.zeros(len(batch), max_obj, 3)
    coord_mask = torch.zeros(len(batch), max_obj, dtype=torch.bool)
    for b, ex in enumerate(batch):
        n = ex.object_coords.shape[0]
        coords[b, :n] = ex.object_coords
        coord_mask[b, :n] = True
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": enc["input_ids"].clone(),
        "object_names": object_names,
        "object_coords": coords,
        "coord_mask": coord_mask,
        "batch": batch,
    }


def train(args: TrainArgs) -> None:
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, wrapper = build_model(args.model_config, args.lora_rank)
    model.train()
    device = next(model.parameters()).device

    dataset = Free6DofDirichletDataset(
        scenes_root=args.train_root, vqa_path=args.train_vqa
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, wrapper.tokenizer),
        num_workers=2,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    dirichlet_loss_fn = DirichletLoss(tau=args.tau, normalize=True)
    layer_module = wrapper.layer_module(args.layer)

    step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Capture the residual stream at the chosen layer
            with ResidualStreamHook(layer_module) as hook:
                out = model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    labels=labels,
                )
                lm_loss = out.loss
                H_all = hook.last  # (B, seq, d)

            # Find object-token positions and gather their activations
            obj_idx, obj_mask = extract_object_token_positions(
                wrapper.tokenizer, input_ids, batch["object_names"]
            )
            obj_idx = obj_idx.to(device)
            obj_mask = obj_mask.to(device) & batch["coord_mask"].to(device)
            # Gather H at those positions: (B, n_obj, d)
            H_obj = torch.gather(
                H_all,
                dim=1,
                index=obj_idx.unsqueeze(-1).expand(-1, -1, H_all.shape[-1]),
            )

            X_obj = batch["object_coords"].to(device)

            # Dirichlet loss (zero contribution where mask is False)
            if obj_mask.any():
                dir_loss = dirichlet_loss_fn(H_obj, X_obj, valid_mask=obj_mask)
            else:
                dir_loss = torch.zeros((), device=device)

            loss = lm_loss + args.lambda_dir * dir_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip
            )
            optimizer.step()

            if step % args.log_every == 0:
                logger.info(
                    "step=%d  lm=%.4f  dir=%.4f  total=%.4f",
                    step, lm_loss.item(), dir_loss.item(), loss.item(),
                )
            if step > 0 and step % args.save_every_steps == 0:
                ckpt = args.output_dir / f"step_{step}"
                model.save_pretrained(ckpt)
                logger.info("saved checkpoint -> %s", ckpt)

            step += 1

    final_ckpt = args.output_dir / "final"
    model.save_pretrained(final_ckpt)
    logger.info("saved final checkpoint -> %s", final_ckpt)


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #


def parse_args() -> TrainArgs:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--model-config", type=Path, required=True)
    p.add_argument("--train-root", type=Path, required=True)
    p.add_argument("--train-vqa", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--layer", type=int, required=True,
                   help="Layer index for the Dirichlet hook (≈ residualized-RSA peak)")
    p.add_argument("--lambda-dir", type=float, default=0.1,
                   help="Weight on the Dirichlet term in the total loss")
    p.add_argument("--tau", type=float, default=1.0,
                   help="Gaussian-kernel bandwidth (units of scene coordinates)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    ns = p.parse_args()
    return TrainArgs(**vars(ns))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )
    train(parse_args())


if __name__ == "__main__":
    main()
