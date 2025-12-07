# light_correction.py

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class LightCorrectionConfig:
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple = (8, 8)
    gamma_target_mean: float = 0.5
    gamma_min: float = 0.7
    gamma_max: float = 1.5
    enable_white_balance: bool = True
    enable_clahe: bool = True
    enable_gamma: bool = True


class LightCorrection:
    """Module to correct lighting issues such as under/over-exposure
    and color casts using white balance, CLAHE, and adaptive gamma."""

    def __init__(self, config: LightCorrectionConfig | None = None):
        if config is None:
            config = LightCorrectionConfig()
        self.config = config

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.correct(img)

    def correct(self, img: np.ndarray) -> np.ndarray:
        """Apply lighting correction to a BGR image (uint8 ndarray)."""
        if img is None:
            raise ValueError("Input image is None")
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Expected a BGR color image with shape (H, W, 3)")

        out = img.copy()
        if self.config.enable_white_balance:
            out = self._gray_world_white_balance(out)
        if self.config.enable_clahe:
            out = self._clahe_l_channel(out)
        if self.config.enable_gamma:
            out = self._auto_gamma(out)
        return out

    def _gray_world_white_balance(self, img: np.ndarray) -> np.ndarray:
        """Simple gray-world white balance to reduce color casts."""
        img_f = img.astype(np.float32)
        b_avg, g_avg, r_avg, _ = cv2.mean(img_f)
        avg_gray = (b_avg + g_avg + r_avg) / 3.0
        scale_b = avg_gray / (b_avg + 1e-6)
        scale_g = avg_gray / (g_avg + 1e-6)
        scale_r = avg_gray / (r_avg + 1e-6)
        img_f[:, :, 0] *= scale_b
        img_f[:, :, 1] *= scale_g
        img_f[:, :, 2] *= scale_r
        img_f = np.clip(img_f, 0, 255)
        return img_f.astype(np.uint8)

    def _clahe_l_channel(self, img: np.ndarray) -> np.ndarray:
        """Apply CLAHE on the L channel in LAB color space for local contrast."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid_size,
        )
        cl = clahe.apply(l)
        lab_clahe = cv2.merge((cl, a, b))
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    def _auto_gamma(self, img: np.ndarray) -> np.ndarray:
        """Adaptively choose gamma so that the output mean brightness
        moves toward gamma_target_mean."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean = gray.mean() / 255.0
        eps = 1e-3
        mean = float(np.clip(mean, eps, 1 - eps))
        target = self.config.gamma_target_mean
        target = float(np.clip(target, eps, 1 - eps))
        gamma = np.log(target) / np.log(mean)
        gamma = float(np.clip(gamma, self.config.gamma_min, self.config.gamma_max))

        img_f = img.astype(np.float32) / 255.0
        img_gamma = np.power(img_f, gamma)
        img_gamma = np.clip(img_gamma * 255.0, 0, 255)
        return img_gamma.astype(np.uint8)