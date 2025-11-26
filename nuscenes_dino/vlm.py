"""Minimal Dino-style image encoder wrapper."""

from collections.abc import Iterable

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class DinoImageEncoder:
    """Adapts a Dino-style vision backbone for nuScenes camera streams."""

    def __init__(self, model_name: str = "facebook/dinov2-base", device: torch.device | str | None = None) -> None:
        """Load the encoder and place it on CPU or GPU.

        Args:
            model_name: Hugging Face identifier for the Dino checkpoint.
            device: Optional torch device or string (e.g., "cuda", "cpu"). If
                omitted, the encoder chooses CUDA when available.
        """

        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, images: Iterable[Image.Image]) -> dict[str, torch.Tensor]:
        """Encode a batch of images and return pooled features.

        The return dict matches the signature used by Hugging Face vision models
        (pixel_values, last_hidden_state, pooled_output) to make it easy to pair
        with a language model or downstream decoder.
        """

        batch: list[Image.Image] = list(images)
        inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        pooled = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state.mean(dim=1)
        return {
            "pixel_values": inputs["pixel_values"],
            "last_hidden_state": outputs.last_hidden_state,
            "pooled_output": pooled,
        }
