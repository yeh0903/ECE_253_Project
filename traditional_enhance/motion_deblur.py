import cv2
import numpy as np
from dataclasses import dataclass
from typing import Sequence, Iterable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


def variance_of_laplacian(img: np.ndarray) -> float:
    """Focus measure based on the variance of the Laplacian.
    Lower values = more blur."""
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def motion_kernel(length: int = 15, angle: float = 0.0) -> np.ndarray:
    """Create a normalized 2D motion blur kernel of given length and angle (degrees)."""
    length = max(3, int(length))
    if length % 2 == 0:
        length += 1

    kernel = np.zeros((length, length), dtype=np.float32)
    cv2.line(kernel, (0, length // 2), (length - 1, length // 2), 1, 1)

    center = (length / 2.0, length / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (length, length))
    s = kernel.sum()
    if s > 0:
        kernel /= s
    return kernel


def richardson_lucy_channel(
    image: np.ndarray,
    psf: np.ndarray,
    iterations: int = 15,
    eps: float = 1e-3,
    early_stop: bool = False,
    tol: float = 1e-4,
    check_every: int = 3,
) -> np.ndarray:
    """Richardson–Lucy deconvolution on a single channel. image: float32 in [0, 1], psf: float32, normalized."""
    if image.ndim != 2:
        raise ValueError("richardson_lucy_channel expects a 2D array.")

    img = np.clip(image.astype(np.float32), 0.0, 1.0)

    psf = psf.astype(np.float32)
    psf /= (psf.sum() + 1e-8)
    psf_mirror = psf[::-1, ::-1]

    latent = img.copy()
    prev_latent = latent.copy()

    for i in range(int(iterations)):
        conv = cv2.filter2D(latent, -1, psf, borderType=cv2.BORDER_REFLECT)
        relative_blur = img / (conv + eps)
        latent *= cv2.filter2D(relative_blur, -1, psf_mirror, borderType=cv2.BORDER_REFLECT)
        latent = np.clip(latent, 0.0, 1.0)

        if early_stop and (i + 1) % check_every == 0:
            diff = np.mean(np.abs(latent - prev_latent))
            if diff < tol:
                break
            prev_latent[:] = latent

    return latent


def richardson_lucy_color(
    img: np.ndarray,
    psf: np.ndarray,
    iterations: int = 15,
) -> np.ndarray:
    """Apply Richardson–Lucy deconvolution per channel on a BGR uint8 image."""
    img_f = img.astype(np.float32) / 255.0
    b, g, r = cv2.split(img_f)
    b_rec = richardson_lucy_channel(b, psf, iterations)
    g_rec = richardson_lucy_channel(g, psf, iterations)
    r_rec = richardson_lucy_channel(r, psf, iterations)
    out = cv2.merge((b_rec, g_rec, r_rec))
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return out


def unsharp_mask(img: np.ndarray, amount: float = 0.4, sigma: float = 1.0) -> np.ndarray:
    """Unsharp masking for local edge enhancement."""
    img_f = img.astype(np.float32)
    sigma = float(sigma)
    ksize = max(3, int(2 * round(3 * sigma) + 1))
    blurred = cv2.GaussianBlur(img_f, (ksize, ksize), sigmaX=sigma)
    sharpened = cv2.addWeighted(img_f, 1 + amount, blurred, -amount, 0)
    sharpened = np.clip(sharpened, 0, 255)
    return sharpened.astype(np.uint8)


@dataclass
class MotionDeblurConfig:
    # Search over multiple PSF lengths and angles
    lengths: Sequence[int] = (9, 15, 21, 31)
    angles: Sequence[float] = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0)

    iterations: int = 15           # RL iterations per final full-res candidate
    blur_threshold: float = 200.0  # if Laplacian variance < this => try deblur
    apply_always: bool = False     # force deblur regardless of sharpness

    pre_denoise: bool = True       # denoise before deblurring
    denoise_h: int = 5
    denoise_h_color: int = 5
    denoise_template_window_size: int = 7
    denoise_search_window_size: int = 21

    final_unsharp_amount: float = 0.5
    final_unsharp_sigma: float = 1.0

    # Preview settings for PSF search (cheap RL on small grayscale)
    preview_scale: float = 0.5
    preview_iterations: int = 5
    sharpness_crop: int = 256

    # Concurrency controls for PSF search
    use_thread_pool: bool = True
    use_process_pool: bool = False  # set True to use ProcessPoolExecutor instead
    max_workers: int | None = None  # None => default number of workers


def _center_crop_gray(img: np.ndarray, crop_size: int) -> np.ndarray:
    """Return a central square crop of the grayscale image."""
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    size = min(crop_size, h, w)
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return img[y0:y0 + size, x0:x0 + size]


def _evaluate_psf_preview(
    gray_f: np.ndarray,
    psf: np.ndarray,
    iterations: int,
    sharpness_crop_size: int,
) -> float:
    """
    Run 1-channel RL on the preview image with a given PSF
    and return a Laplacian-variance sharpness score.
    This function is used in thread/process pools.
    """
    candidate = richardson_lucy_channel(
        gray_f,
        psf,
        iterations=iterations,
    )
    cand_u8 = np.clip(candidate * 255.0, 0, 255).astype(np.uint8)
    crop = _center_crop_gray(cand_u8, sharpness_crop_size)
    score = variance_of_laplacian(crop)
    return float(score)


class MotionDeblur:
    """Stronger motion deblurring using multi-kernel Richardson–Lucy with concurrency."""

    def __init__(self, config: MotionDeblurConfig | None = None):
        if config is None:
            config = MotionDeblurConfig()
        self.config = config

        # Precompute PSFs for all (length, angle) pairs
        self.psfs: dict[tuple[int, float], np.ndarray] = {}
        for L in self.config.lengths:
            for A in self.config.angles:
                self.psfs[(L, A)] = motion_kernel(L, A).astype(np.float32)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.deblur(img)

    def _sharpness_crop(self, img: np.ndarray) -> float:
        """Compute Laplacian variance on a central crop to save time."""
        crop = _center_crop_gray(img, self.config.sharpness_crop)
        return variance_of_laplacian(crop)

    def _choose_best_psf(self, base_bgr: np.ndarray) -> tuple[np.ndarray | None, float]:
        """
        Run a cheap RL search on a downscaled grayscale preview
        to pick the best PSF. Returns (best_psf, best_score_delta).
        """
        # Grayscale + downscale
        gray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)
        if self.config.preview_scale < 1.0:
            gray = cv2.resize(
                gray,
                dsize=None,
                fx=self.config.preview_scale,
                fy=self.config.preview_scale,
                interpolation=cv2.INTER_AREA,
            )

        gray_f = gray.astype(np.float32) / 255.0
        orig_score = self._sharpness_crop(gray)

        best_psf: np.ndarray | None = None
        best_score = orig_score

        items = list(self.psfs.items())

        # Decide which executor to use
        ExecutorCls = None
        if self.config.use_process_pool:
            ExecutorCls = ProcessPoolExecutor
        elif self.config.use_thread_pool:
            ExecutorCls = ThreadPoolExecutor

        # Parallel or sequential evaluation
        if ExecutorCls is not None:
            with ExecutorCls(max_workers=self.config.max_workers) as ex:
                futures = []
                for (L, A), psf in items:
                    futures.append(
                        ex.submit(
                            _evaluate_psf_preview,
                            gray_f,
                            psf,
                            self.config.preview_iterations,
                            self.config.sharpness_crop,
                        )
                    )
                for (key, _), fut in zip(items, futures):
                    score = fut.result()
                    if score > best_score:
                        best_score = score
                        best_psf = self.psfs[key]
        else:
            # fallback
            for (L, A), psf in items:
                score = _evaluate_psf_preview(
                    gray_f,
                    psf,
                    self.config.preview_iterations,
                    self.config.sharpness_crop,
                )
                if score > best_score:
                    best_score = score
                    best_psf = psf

        return best_psf, (best_score - orig_score)

    def deblur(self, img: np.ndarray) -> np.ndarray:
        """Deblur a BGR image (uint8 ndarray)."""
        if img is None:
            raise ValueError("Input image is None")
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Expected a BGR color image with shape (H, W, 3)")

        orig_score = self._sharpness_crop(img)
        if not self.config.apply_always and orig_score >= self.config.blur_threshold:
            # Already sharp enough
            return img

        # Optional pre-denoising to avoid amplifying noise
        if self.config.pre_denoise:
            base = cv2.fastNlMeansDenoisingColored(
                img,
                None,
                h=self.config.denoise_h,
                hColor=self.config.denoise_h_color,
                templateWindowSize=self.config.denoise_template_window_size,
                searchWindowSize=self.config.denoise_search_window_size,
            )
        else:
            base = img

        # 1) Pick best PSF on a small grayscale preview (possibly in parallel)
        best_psf, delta = self._choose_best_psf(base)

        # If nothing improved on preview, bail out early
        if best_psf is None:
            return img

        # 2) Run full-res color RL ONCE with the chosen PSF
        candidate = richardson_lucy_color(base, best_psf, self.config.iterations)
        candidate = unsharp_mask(
            candidate,
            amount=self.config.final_unsharp_amount,
            sigma=self.config.final_unsharp_sigma,
        )

        # Optional: only accept if full-res sharpness is better
        final_score = self._sharpness_crop(candidate)
        if final_score > orig_score:
            return candidate
        else:
            return img

def deblur_images_threadpool(
    images: Sequence[np.ndarray],
    config: MotionDeblurConfig | None = None,
    max_workers: int | None = None,
) -> list[np.ndarray]:
    """
    Deblur a sequence of images concurrently using a shared MotionDeblur and threads.
    Good when using OpenCV (releases GIL) and many images.
    """
    if config is None:
        config = MotionDeblurConfig()
    deblurrer = MotionDeblur(config)

    def _worker(img: np.ndarray) -> np.ndarray:
        return deblurrer.deblur(img)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_worker, images))
    return results


def _processpool_worker(args: tuple[np.ndarray, MotionDeblurConfig]) -> np.ndarray:
    """
    Worker function for process pool. Defined at top level so it is picklable.
    Creates its own MotionDeblur per process.
    """
    img, cfg = args
    deblurrer = MotionDeblur(cfg)
    return deblurrer.deblur(img)


def deblur_images_processpool(
    images: Sequence[np.ndarray],
    config: MotionDeblurConfig | None = None,
    max_workers: int | None = None,
) -> list[np.ndarray]:
    """
    Deblur a sequence of images concurrently using a ProcessPoolExecutor.
    This can help for very heavy CPU-bound workloads, at the cost of
    higher overhead and data pickling between processes.
    """
    if config is None:
        config = MotionDeblurConfig()

    # Pack (image, config) pairs so each worker can reconstruct the deblurrer
    payload = [(img, config) for img in images]

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_processpool_worker, payload))
    return results