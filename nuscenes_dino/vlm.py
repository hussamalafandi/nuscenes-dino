"""Minimal Dino-style image encoder wrapper."""

from typing import Dict, Iterable, List

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class DinoImageEncoder:
    """Adapts a Dino-style vision backbone for nuScenes camera streams."""

    def __init__(self, model_name: str = "facebook/dinov2-base") -> None:
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, images: Iterable[Image.Image]) -> Dict[str, torch.Tensor]:
        """Encode a batch of images and return pooled features.

        The return dict matches the signature used by Hugging Face vision models
        (pixel_values, last_hidden_state, pooled_output) to make it easy to pair
        with a language model or downstream decoder.
        """

        batch: List[Image.Image] = list(images)
        inputs = self.processor(images=batch, return_tensors="pt")
        outputs = self.model(**inputs)
        pooled = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state.mean(dim=1)
        return {
            "pixel_values": inputs["pixel_values"],
            "last_hidden_state": outputs.last_hidden_state,
            "pooled_output": pooled,
        }
