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
    ):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self._torch = torch
        dtype = getattr(torch, torch_dtype)
        self.processor = AutoProcessor.from_pretrained(hf_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            hf_id, torch_dtype=dtype, device_map=device
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

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

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
        with torch.no_grad():
            _ = self.model(**proc_inputs, return_dict=True)

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

        return ForwardOut(
            hidden_states=list(self._layer_outputs),
            visual_token_range=(start, end),
            grid=grid,
            extras={"input_ids": input_ids.cpu(), "is_video": is_video},
        )
