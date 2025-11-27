"""Minimal Dino-style image encoder wrapper."""

from collections.abc import Iterable

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class DinoImageEncoder:
    """Adapts a DINOv3 vision backbone for nuScenes camera streams."""

    def __init__(
        self,
        model_name: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        device: torch.device | str | None = None,
        device_map: str | dict[str, int] | None = "auto",
    ) -> None:
        """Load the encoder and place it on CPU or GPU.

        Args:
            model_name: Hugging Face identifier for the DINOv3 checkpoint.
            device: Optional torch device or string (e.g., "cuda", "cpu"). If
                omitted, the encoder chooses CUDA when available unless
                ``device_map`` is provided.
            device_map: Optional device map passed to ``AutoModel.from_pretrained``
                (e.g., ``"auto"``). When set, the model will use this mapping and
                input tensors will follow the model's device placement.
        """

        if device is not None and device_map is not None:
            msg = "Only one of `device` or `device_map` can be provided."
            raise ValueError(msg)

        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_map = device_map

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, device_map=self.device_map)
        if self.device_map is None:
            self.model = self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, images: Iterable[Image.Image]) -> dict[str, torch.Tensor]:
        """Encode a batch of images and return pooled features.

        The return dict matches the signature used by Hugging Face vision models
        (pixel_values, last_hidden_state, pooled_output) to make it easy to pair
        with a language model or downstream decoder.
        """

        batch: list[Image.Image] = list(images)
        inputs = self.processor(images=batch, return_tensors="pt")
        if self.device_map is None:
            inputs = inputs.to(self.device)
        elif hasattr(self.model, "device"):
            inputs = inputs.to(self.model.device)

        outputs = self.model(**inputs)
        pooled = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state.mean(dim=1)
        return {
            "pixel_values": inputs["pixel_values"],
            "last_hidden_state": outputs.last_hidden_state,
            "pooled_output": pooled,
        }
