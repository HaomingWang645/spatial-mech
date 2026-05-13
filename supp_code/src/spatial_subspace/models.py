"""VLM wrappers with hidden-state hook support.

Each wrapper exposes a uniform interface: ``forward(image, prompt)`` returns a
``ForwardOut`` holding the per-layer post-layer hidden states, the slice of
the token sequence that corresponds to visual tokens, and the visual-token
grid shape. The extraction pipeline consumes this directly.

Only Qwen2.5-VL is implemented here. Adding LLaVA-Video / InternVL3 means
writing another subclass against the same interface (the extraction pipeline
in ``extract.py`` is model-agnostic as long as the interface is honoured).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ForwardOut:
    hidden_states: list[Any]              # [num_layers] tensors of shape (B, T, D)
    visual_token_range: tuple[int, int]   # [start, end) in sequence dimension
    grid: tuple[int, int, int]            # (T_video, H_tok, W_tok) post-merger grid
    extras: dict[str, Any] = field(default_factory=dict)


class VLMWrapper(Protocol):
    def forward(self, image: Any, prompt: str) -> ForwardOut: ...
    def patch_pixels(self) -> int: ...
    def image_input_hw(self, out: ForwardOut) -> tuple[int, int]: ...
    def close(self) -> None: ...


class Qwen25VLWrapper:
    """transformers-based wrapper for Qwen2.5-VL-*.

    Hooks every language-model decoder layer. The visual-token slice is
    located by matching the image-token id in ``input_ids``; the grid shape
    comes from the processor's ``image_grid_thw`` corrected by the spatial
    merge size.
    """

    def __init__(
        self,
        hf_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype: str = "bfloat16",
        device: str = "cuda",
        device_map: str | None = None,
    ):
        """``device`` is where input tensors live; ``device_map`` (if given)
        is passed to ``from_pretrained`` for model sharding. For a single-GPU
        load use device="cuda" and leave device_map unset. For multi-GPU
        sharding use device="cuda" and device_map="auto" (or a dict)."""
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self._torch = torch
        dtype = getattr(torch, torch_dtype)
        self.processor = AutoProcessor.from_pretrained(hf_id)
        load_map = device_map if device_map is not None else device
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            hf_id, torch_dtype=dtype, device_map=load_map
        )
        self.model.eval()
        self.device = device

        self._image_token_id = getattr(self.model.config, "image_token_id", None)
        if self._image_token_id is None:
            self._image_token_id = self.processor.tokenizer.convert_tokens_to_ids(
                "<|image_pad|>"
            )
        self._video_token_id = getattr(self.model.config, "video_token_id", None)
        if self._video_token_id is None:
            self._video_token_id = self.processor.tokenizer.convert_tokens_to_ids(
                "<|video_pad|>"
            )

        self._layer_outputs: list[Any] = []
        self._handles: list[Any] = []
        # Per-head capture state (opt-in via enable_head_capture).
        self._head_inputs: dict[int, Any] = {}
        self._head_handles: list[Any] = []
        self._head_layers: list[int] = []
        self._register_hooks()

    def _locate_layers(self):
        # Cover both the flat layout (older transformers) and the nested
        # Qwen2_5_VLModel → Qwen2_5_VLTextModel → layers layout introduced
        # around transformers 4.50.
        for path in [
            ("model", "language_model", "layers"),
            ("language_model", "model", "layers"),
            ("model", "model", "layers"),
            ("model", "layers"),
            ("language_model", "layers"),
        ]:
            obj = self.model
            ok = True
            for attr in path:
                if not hasattr(obj, attr):
                    ok = False
                    break
                obj = getattr(obj, attr)
            if ok:
                return obj
        raise RuntimeError("could not locate decoder layers on this Qwen2.5-VL build")

    def _register_hooks(self) -> None:
        layers = self._locate_layers()

        def make_hook(_idx: int):
            def hook(_module, _inputs, output):
                hs = output[0] if isinstance(output, tuple) else output
                self._layer_outputs.append(hs.detach())
            return hook

        for i, layer in enumerate(layers):
            self._handles.append(layer.register_forward_hook(make_hook(i)))

    def enable_head_capture(self, layer_ids: list[int]) -> None:
        """Register pre-hooks on ``self_attn.o_proj`` of each given layer that
        capture the projection's input — the concatenated multi-head output of
        shape ``(B, T, n_heads * head_dim)``. Each forward pass populates
        ``self._head_inputs`` as ``{layer_idx: tensor(B, T, n_heads * head_dim)}``.

        Combined with ``o_proj_weight(layer_idx)``, this gives every head's
        additive contribution to the residual stream at that layer:
            c_h = x[:, :, h*d:(h+1)*d] @ W_O[:, h*d:(h+1)*d].T
        summed over h plus bias equals ``o_proj(x)``.
        """
        self.disable_head_capture()
        layers = self._locate_layers()

        def make_pre_hook(idx: int):
            def pre_hook(_mod, args, kwargs):
                x = args[0] if args else kwargs.get("input")
                if x is not None:
                    self._head_inputs[idx] = x.detach()
                return None
            return pre_hook

        self._head_layers = list(layer_ids)
        for idx in layer_ids:
            o_proj = layers[idx].self_attn.o_proj
            self._head_handles.append(
                o_proj.register_forward_pre_hook(make_pre_hook(idx), with_kwargs=True)
            )

    def disable_head_capture(self) -> None:
        for h in self._head_handles:
            h.remove()
        self._head_handles.clear()
        self._head_inputs.clear()
        self._head_layers.clear()

    def o_proj_weight(self, layer_idx: int):
        """Return ``o_proj.weight`` (shape (D, n_heads * head_dim)) for the
        given layer. Caller is responsible for slicing per-head columns.
        """
        layers = self._locate_layers()
        return layers[layer_idx].self_attn.o_proj.weight

    def attn_head_dims(self) -> tuple[int, int]:
        """Return ``(n_heads, head_dim)`` for the language backbone."""
        cfg = self.model.config
        tc = getattr(cfg, "text_config", None) or cfg
        n_heads = int(tc.num_attention_heads)
        head_dim = int(getattr(tc, "head_dim", tc.hidden_size // n_heads))
        return n_heads, head_dim

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.disable_head_capture()

    def install_intervention(
        self,
        layer_idx: int,
        token_positions: list[int],
        delta,
    ):
        """Register a forward_pre_hook on layer ``layer_idx`` that adds
        ``delta`` (a 1-D tensor of shape (D,)) to the hidden states at each
        position in ``token_positions`` before the layer executes. Returns a
        handle whose ``.remove()`` undoes the intervention.

        pre-hook so the perturbation enters the residual stream BEFORE the
        layer transforms it, matching the plan §6.1 "add Δ to h at layer L"
        semantics. The capture hooks registered in ``_register_hooks`` fire on
        the layer's output, so they will record the post-intervention
        activations for downstream analysis.
        """
        torch = self._torch
        layers = self._locate_layers()
        target_layer = layers[layer_idx]
        if not isinstance(delta, torch.Tensor):
            delta = torch.as_tensor(delta)
        positions = list(token_positions)

        def pre_hook(_mod, args, kwargs):
            if args:
                hs = args[0]
            elif "hidden_states" in kwargs:
                hs = kwargs["hidden_states"]
            else:
                return None
            d = delta.to(device=hs.device, dtype=hs.dtype)
            if positions:
                idx = torch.as_tensor(positions, device=hs.device, dtype=torch.long)
                hs[:, idx, :] = hs[:, idx, :] + d
            if args:
                return (hs, *args[1:]), kwargs
            kwargs["hidden_states"] = hs
            return args, kwargs

        return target_layer.register_forward_pre_hook(pre_hook, with_kwargs=True)

    def patch_pixels(self) -> int:
        vc = self.model.config.vision_config
        ps = int(getattr(vc, "patch_size", 14))
        merge = int(getattr(vc, "spatial_merge_size", 2))
        return ps * merge

    def temporal_patch_size(self) -> int:
        return int(getattr(self.model.config.vision_config, "temporal_patch_size", 2))

    def image_input_hw(self, out: ForwardOut) -> tuple[int, int]:
        _, h_tok, w_tok = out.grid
        pp = self.patch_pixels()
        return h_tok * pp, w_tok * pp

    def forward(self, image: Any, prompt: str) -> ForwardOut:
        """Run a forward pass and capture per-layer hidden states.

        ``image`` may be a single PIL/path/URI (image pathway) or a list of
        them (video pathway). The video pathway feeds frames through
        ``process_vision_info`` so the model sees them with M-RoPE temporal
        positions and applies its temporal patch merger.
        """
        torch = self._torch
        from qwen_vl_utils import process_vision_info

        is_video = isinstance(image, (list, tuple))
        if is_video:
            content = [
                {"type": "video", "video": list(image), "fps": 1.0},
                {"type": "text", "text": prompt},
            ]
            target_token_id = self._video_token_id
            grid_key = "video_grid_thw"
        else:
            content = [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]
            target_token_id = self._image_token_id
            grid_key = "image_grid_thw"

        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        proc_inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        self._layer_outputs.clear()
        self._head_inputs.clear()
        with torch.no_grad():
            model_out = self.model(**proc_inputs, return_dict=True)

        input_ids = proc_inputs["input_ids"][0]
        positions = (input_ids == target_token_id).nonzero(as_tuple=True)[0]
        if positions.numel() == 0:
            kind = "video" if is_video else "image"
            raise RuntimeError(f"no {kind} tokens found in input_ids")
        start = int(positions[0].item())
        end = int(positions[-1].item()) + 1

        grid_thw = proc_inputs.get(grid_key)
        if grid_thw is None:
            raise RuntimeError(f"processor did not return {grid_key}")
        t, h, w = [int(x) for x in grid_thw[0].tolist()]
        merge = int(getattr(self.model.config.vision_config, "spatial_merge_size", 2))
        # For images: t is 1 (no temporal merging applies). For videos: t is
        # already post-temporal-merge (video_grid_thw[0] = n_input_frames /
        # temporal_patch_size). So in both cases the LM sees t * (h/merge) *
        # (w/merge) visual tokens.
        grid = (t, h // merge, w // merge)
        expected = t * (h // merge) * (w // merge)
        if (end - start) != expected:
            raise RuntimeError(
                f"visual token count mismatch: range={end - start} vs grid={expected}"
            )

        extras = {"input_ids": input_ids.cpu(), "is_video": is_video}
        if getattr(model_out, "logits", None) is not None:
            extras["logits_last"] = model_out.logits[0, -1, :].detach().float().cpu().numpy()
        if self._head_inputs:
            # Move captured per-layer head inputs into extras; leave GPU residency
            # as-is so callers can slice-then-project on the right device.
            extras["head_inputs"] = dict(self._head_inputs)
        return ForwardOut(
            hidden_states=list(self._layer_outputs),
            visual_token_range=(start, end),
            grid=grid,
            extras=extras,
        )


# ---------------------------------------------------------------------------
# LLaVA-OneVision
# ---------------------------------------------------------------------------


class LlavaOnevisionWrapper:
    """transformers-based wrapper for LLaVA-OneVision-*.

    Video input:
        frames → 384×384 ViT (patch 14) → 27×27 patches → 2×2 spatial pool
        → 14×14 tokens per frame. All N frames are kept (no temporal merger),
        producing a contiguous run of ``N * 14 * 14`` ``<video>`` tokens plus
        one trailing newline. We slice that trailing token away to keep the
        visual-token slice an exact multiple of 14×14.
    """

    def __init__(
        self,
        hf_id: str = "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        torch_dtype: str = "bfloat16",
        device: str = "cuda",
        device_map: str | None = None,
    ):
        import torch
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

        self._torch = torch
        dtype = getattr(torch, torch_dtype)
        self.processor = AutoProcessor.from_pretrained(hf_id)
        load_map = device_map if device_map is not None else device
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            hf_id, torch_dtype=dtype, device_map=load_map
        )
        self.model.eval()
        self.device = device

        tok = self.processor.tokenizer
        self._video_token_id = tok.convert_tokens_to_ids("<video>")
        self._image_token_id = tok.convert_tokens_to_ids("<image>")

        # Vision geometry: SigLIP 384×384 patch 14 → 27×27; 2×2 pool → 14×14
        vc = self.model.config.vision_config
        self._vit_patch = int(getattr(vc, "patch_size", 14))
        self._vit_image_size = int(getattr(vc, "image_size", 384))
        # Per-frame token grid after the 2×2 spatial pool
        per_side_pre = self._vit_image_size // self._vit_patch
        self._tok_side = (per_side_pre + 1) // 2  # 27 → 14
        # Each output token covers a 2 × vit_patch block in the raw image
        self._patch_pixels = 2 * self._vit_patch

        self._layer_outputs: list[Any] = []
        self._handles: list[Any] = []
        self._register_hooks()

    def _locate_layers(self):
        # LlavaOnevision uses a Qwen2-based LM nested under `.language_model`.
        for path in [
            ("model", "language_model", "layers"),
            ("language_model", "model", "layers"),
            ("model", "language_model", "model", "layers"),
            ("language_model", "layers"),
        ]:
            obj = self.model
            ok = True
            for attr in path:
                if not hasattr(obj, attr):
                    ok = False
                    break
                obj = getattr(obj, attr)
            if ok:
                return obj
        raise RuntimeError("could not locate decoder layers on this LLaVA-OV build")

    def _register_hooks(self) -> None:
        layers = self._locate_layers()

        def make_hook(_idx: int):
            def hook(_module, _inputs, output):
                hs = output[0] if isinstance(output, tuple) else output
                self._layer_outputs.append(hs.detach())
            return hook

        for i, layer in enumerate(layers):
            self._handles.append(layer.register_forward_hook(make_hook(i)))

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def install_intervention(
        self,
        layer_idx: int,
        token_positions: list[int],
        delta,
    ):
        torch = self._torch
        layers = self._locate_layers()
        target_layer = layers[layer_idx]
        if not isinstance(delta, torch.Tensor):
            delta = torch.as_tensor(delta)
        positions = list(token_positions)

        def pre_hook(_mod, args, kwargs):
            if args:
                hs = args[0]
            elif "hidden_states" in kwargs:
                hs = kwargs["hidden_states"]
            else:
                return None
            d = delta.to(device=hs.device, dtype=hs.dtype)
            if positions:
                idx = torch.as_tensor(positions, device=hs.device, dtype=torch.long)
                hs[:, idx, :] = hs[:, idx, :] + d
            if args:
                return (hs, *args[1:]), kwargs
            kwargs["hidden_states"] = hs
            return args, kwargs

        return target_layer.register_forward_pre_hook(pre_hook, with_kwargs=True)

    def patch_pixels(self) -> int:
        return self._patch_pixels

    def temporal_patch_size(self) -> int:
        return 1  # LLaVA-OV keeps every input frame as one temporal slot

    def image_input_hw(self, out: ForwardOut) -> tuple[int, int]:
        return self._vit_image_size, self._vit_image_size

    def forward(self, image: Any, prompt: str) -> ForwardOut:
        torch = self._torch
        is_video = isinstance(image, (list, tuple))

        # Load frames as PIL; the processor's video path expects a list of PIL images.
        from PIL import Image
        def _load(x):
            if isinstance(x, str) and x.startswith("file://"):
                return Image.open(x[len("file://"):]).convert("RGB")
            if isinstance(x, str):
                return Image.open(x).convert("RGB")
            return x

        if is_video:
            frames = [_load(f) for f in image]
            content = [{"type": "video"}, {"type": "text", "text": prompt}]
            target_token_id = self._video_token_id
        else:
            frames = _load(image)
            content = [{"type": "image"}, {"type": "text", "text": prompt}]
            target_token_id = self._image_token_id

        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        proc_kwargs: dict[str, Any] = {"text": [text], "return_tensors": "pt"}
        if is_video:
            proc_kwargs["videos"] = [frames]
        else:
            proc_kwargs["images"] = [frames]
        proc_inputs = self.processor(**proc_kwargs).to(self.device)

        self._layer_outputs.clear()
        with torch.no_grad():
            model_out = self.model(**proc_inputs, return_dict=True)

        input_ids = proc_inputs["input_ids"][0]
        mask = input_ids == target_token_id
        n_visual = int(mask.sum().item())
        if n_visual == 0:
            kind = "video" if is_video else "image"
            raise RuntimeError(f"no {kind} tokens found in input_ids")

        # LLaVA-OV appends one trailing newline inside the <video> run for
        # video inputs. Strip it so the surviving count equals T*H*W exactly.
        positions = mask.nonzero(as_tuple=True)[0]
        side = self._tok_side
        per_frame = side * side
        if is_video:
            # Number of input frames supplied
            if "pixel_values_videos" in proc_inputs:
                t_input = int(proc_inputs["pixel_values_videos"].shape[1])
            else:
                t_input = n_visual // per_frame
            expected = t_input * per_frame
            if n_visual not in (expected, expected + 1):
                raise RuntimeError(
                    f"unexpected visual token count: got {n_visual}, "
                    f"expected {expected} or {expected + 1}"
                )
            grid = (t_input, side, side)
            positions = positions[:expected]  # drop the trailing newline
        else:
            # Single image: n_visual should be side*side (no +1)
            grid = (1, side, side)
            positions = positions[:per_frame]

        start = int(positions[0].item())
        end = int(positions[-1].item()) + 1

        extras: dict[str, Any] = {
            "input_ids": input_ids.cpu(),
            "is_video": is_video,
        }
        if getattr(model_out, "logits", None) is not None:
            extras["logits_last"] = model_out.logits[0, -1, :].detach().float().cpu().numpy()
        # Positions are not strictly required in LLaVA-OV's video case (it's
        # contiguous once the trailing newline is dropped), but recording them
        # lets extract.py handle any future non-contiguous variants cleanly.
        extras["visual_positions"] = positions.detach().cpu().numpy()
        return ForwardOut(
            hidden_states=list(self._layer_outputs),
            visual_token_range=(start, end),
            grid=grid,
            extras=extras,
        )


# ---------------------------------------------------------------------------
# InternVL3
# ---------------------------------------------------------------------------


class InternVL3Wrapper:
    """transformers-based wrapper for InternVL3-*.

    Video input:
        frames → 448×448 ViT (patch 14) → 32×32 patches → 2×2 pixel-shuffle
        → 16×16 tokens per frame. Each frame is wrapped in ``<img>...</img>``
        markers, so the N * 256 ``<IMG_CONTEXT>`` tokens are NOT contiguous —
        they come in N runs of 256 separated by single marker tokens. We
        gather all IMG_CONTEXT positions into a 1-D index array and hand it
        to the extract pipeline via ``extras['visual_positions']``.
    """

    def __init__(
        self,
        hf_id: str = "OpenGVLab/InternVL3-8B-hf",
        torch_dtype: str = "bfloat16",
        device: str = "cuda",
        device_map: str | None = None,
    ):
        import torch
        from transformers import AutoProcessor, InternVLForConditionalGeneration

        self._torch = torch
        dtype = getattr(torch, torch_dtype)
        self.processor = AutoProcessor.from_pretrained(hf_id)
        load_map = device_map if device_map is not None else device
        self.model = InternVLForConditionalGeneration.from_pretrained(
            hf_id, torch_dtype=dtype, device_map=load_map
        )
        self.model.eval()
        self.device = device

        tok = self.processor.tokenizer
        self._ctx_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")

        vc = self.model.config.vision_config
        raw_ps = getattr(vc, "patch_size", 14)
        self._vit_patch = int(raw_ps[0] if isinstance(raw_ps, (list, tuple)) else raw_ps)
        raw_is = getattr(vc, "image_size", 448)
        self._vit_image_size = int(raw_is[0] if isinstance(raw_is, (list, tuple)) else raw_is)
        # 2×2 pixel shuffle halves each dim: 32×32 → 16×16 (for 448/14=32)
        self._tok_side = (self._vit_image_size // self._vit_patch) // 2
        self._patch_pixels = 2 * self._vit_patch

        self._layer_outputs: list[Any] = []
        self._handles: list[Any] = []
        self._register_hooks()

    def _locate_layers(self):
        for path in [
            ("model", "language_model", "layers"),
            ("language_model", "model", "layers"),
            ("model", "language_model", "model", "layers"),
            ("language_model", "layers"),
        ]:
            obj = self.model
            ok = True
            for attr in path:
                if not hasattr(obj, attr):
                    ok = False
                    break
                obj = getattr(obj, attr)
            if ok:
                return obj
        raise RuntimeError("could not locate decoder layers on this InternVL3 build")

    def _register_hooks(self) -> None:
        layers = self._locate_layers()

        def make_hook(_idx: int):
            def hook(_module, _inputs, output):
                hs = output[0] if isinstance(output, tuple) else output
                self._layer_outputs.append(hs.detach())
            return hook

        for i, layer in enumerate(layers):
            self._handles.append(layer.register_forward_hook(make_hook(i)))

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def install_intervention(self, layer_idx, token_positions, delta):
        torch = self._torch
        layers = self._locate_layers()
        target_layer = layers[layer_idx]
        if not isinstance(delta, torch.Tensor):
            delta = torch.as_tensor(delta)
        positions = list(token_positions)

        def pre_hook(_mod, args, kwargs):
            if args:
                hs = args[0]
            elif "hidden_states" in kwargs:
                hs = kwargs["hidden_states"]
            else:
                return None
            d = delta.to(device=hs.device, dtype=hs.dtype)
            if positions:
                idx = torch.as_tensor(positions, device=hs.device, dtype=torch.long)
                hs[:, idx, :] = hs[:, idx, :] + d
            if args:
                return (hs, *args[1:]), kwargs
            kwargs["hidden_states"] = hs
            return args, kwargs

        return target_layer.register_forward_pre_hook(pre_hook, with_kwargs=True)

    def patch_pixels(self) -> int:
        return self._patch_pixels

    def temporal_patch_size(self) -> int:
        return 1

    def image_input_hw(self, out: ForwardOut) -> tuple[int, int]:
        return self._vit_image_size, self._vit_image_size

    def forward(self, image: Any, prompt: str) -> ForwardOut:
        torch = self._torch
        is_video = isinstance(image, (list, tuple))

        from PIL import Image
        def _load(x):
            if isinstance(x, str) and x.startswith("file://"):
                return Image.open(x[len("file://"):]).convert("RGB")
            if isinstance(x, str):
                return Image.open(x).convert("RGB")
            return x

        if is_video:
            frames = [_load(f) for f in image]
            content = [{"type": "video"}, {"type": "text", "text": prompt}]
        else:
            frames = _load(image)
            content = [{"type": "image"}, {"type": "text", "text": prompt}]

        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        proc_kwargs: dict[str, Any] = {"text": [text], "return_tensors": "pt"}
        if is_video:
            proc_kwargs["videos"] = [frames]
        else:
            proc_kwargs["images"] = [frames]
        proc_inputs = self.processor(**proc_kwargs).to(self.device)

        self._layer_outputs.clear()
        with torch.no_grad():
            model_out = self.model(**proc_inputs, return_dict=True)

        input_ids = proc_inputs["input_ids"][0]
        mask = input_ids == self._ctx_token_id
        positions = mask.nonzero(as_tuple=True)[0]
        n_visual = int(positions.numel())
        if n_visual == 0:
            kind = "video" if is_video else "image"
            raise RuntimeError(f"no {kind} tokens found in input_ids")

        side = self._tok_side
        per_frame = side * side
        if n_visual % per_frame != 0:
            raise RuntimeError(
                f"visual token count {n_visual} not divisible by per-frame {per_frame}"
            )
        t_input = n_visual // per_frame
        grid = (t_input, side, side)

        # Non-contiguous: we hand positions directly to the extract pipeline.
        start = int(positions[0].item())
        end = int(positions[-1].item()) + 1

        extras: dict[str, Any] = {
            "input_ids": input_ids.cpu(),
            "is_video": is_video,
            "visual_positions": positions.detach().cpu().numpy(),
        }
        if getattr(model_out, "logits", None) is not None:
            extras["logits_last"] = model_out.logits[0, -1, :].detach().float().cpu().numpy()
        return ForwardOut(
            hidden_states=list(self._layer_outputs),
            visual_token_range=(start, end),
            grid=grid,
            extras=extras,
        )
