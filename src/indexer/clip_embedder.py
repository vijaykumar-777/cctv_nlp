"""CLIP embedder — encodes images and text into shared vector space."""

from __future__ import annotations

import numpy as np
import open_clip
import torch
from PIL import Image

from src.config import settings


class CLIPEmbedder:
    """Embeds images and text using OpenCLIP.

    Loads the model once on init.  Both embed_image and embed_text
    return L2-normalised float32 vectors of shape (512,).
    """

    def __init__(self) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        model, _, preprocess = open_clip.create_model_and_transforms(
            settings.clip_model_name, pretrained=settings.clip_pretrained
        )
        self._model = model.to(self._device)
        self._model.eval()
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(settings.clip_model_name)

    def embed_image(self, frame_crop: np.ndarray) -> np.ndarray:
        """Encode a BGR numpy crop into a 512-d normalised vector."""
        # Convert BGR (OpenCV) → RGB → PIL
        rgb = frame_crop[:, :, ::-1] if frame_crop.ndim == 3 else frame_crop
        pil_img = Image.fromarray(rgb)

        tensor = self._preprocess(pil_img).unsqueeze(0).to(self._device)

        with torch.no_grad():
            features = self._model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.squeeze(0).cpu().numpy().astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        """Encode a text string into a 512-d normalised vector."""
        tokens = self._tokenizer([text]).to(self._device)

        with torch.no_grad():
            features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.squeeze(0).cpu().numpy().astype(np.float32)
