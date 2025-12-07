# quality_enhancement.py

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class QualityEnhancementConfig:
    target_min_side: int = 640
    upscale_interpolation: int = cv2.INTER_CUBIC
    denoise_h: int = 10
    denoise_h_color: int = 10
    denoise_template_window_size: int = 7
    denoise_search_window_size: int = 21
    unsharp_amount: float = 0.7
    unsharp_sigma: float = 1.0


class QualityEnhancer:
    """Module to enhance low-quality images (e.g., downsampled or re-screenshotted)
    using upscaling, denoising, and unsharp masking."""

    def __init__(self, config: QualityEnhancementConfig | None = None):
        if config is None:
            config = QualityEnhancementConfig()
        self.config = config

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.enhance(img)

    def enhance(self, img: np.ndarray) -> np.ndarray:
        """Enhance a BGR image (uint8 ndarray)."""
        if img is None:
            raise ValueError("Input image is None")
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Expected a BGR color image with shape (H, W, 3)")

        out = self._upscale_if_needed(img)
        out = self._denoise(out)
        out = self._unsharp_mask(out)
        return out

    def _upscale_if_needed(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        min_side = min(h, w)
        if min_side >= self.config.target_min_side:
            return img
        scale = self.config.target_min_side / float(min_side)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return cv2.resize(img, (new_w, new_h), interpolation=self.config.upscale_interpolation)

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        """Use non-local means denoising to reduce compression artifacts and noise."""
        return cv2.fastNlMeansDenoisingColored(
            img,
            None,
            h=self.config.denoise_h,
            hColor=self.config.denoise_h_color,
            templateWindowSize=self.config.denoise_template_window_size,
            searchWindowSize=self.config.denoise_search_window_size,
        )

    def _unsharp_mask(self, img: np.ndarray) -> np.ndarray:
        """Apply unsharp masking for edge enhancement."""
        img_f = img.astype(np.float32)
        # Automatically choose kernel size based on sigma
        sigma = self.config.unsharp_sigma
        ksize = max(3, int(2 * round(3 * sigma) + 1))
        blurred = cv2.GaussianBlur(img_f, (ksize, ksize), sigmaX=sigma)
        amount = self.config.unsharp_amount
        sharpened = cv2.addWeighted(img_f, 1 + amount, blurred, -amount, 0)
        sharpened = np.clip(sharpened, 0, 255)
        return sharpened.astype(np.uint8)