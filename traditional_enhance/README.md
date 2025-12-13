# Traditional Image Processing

A collection of classical computer vision techniques for restoring and enhancing degraded images. This project implements three core modules for addressing common image quality issues without deep learning.

## Overview

This toolkit provides standalone modules for:
- **Quality Enhancement**: Restore low-quality, downsampled, or re-screenshotted images
- **Motion Deblurring**: Remove motion blur using Richardson-Lucy deconvolution
- **Light Correction**: Fix exposure problems and color casts

## Modules

### Quality Enhancement (`quality_enhancement.py`)

Enhances low-quality images through a three-stage pipeline:

1. **Upscaling**: Resizes small images using cubic interpolation
2. **Denoising**: Applies non-local means denoising to reduce compression artifacts
3. **Sharpening**: Uses unsharp masking for edge enhancement

**Key Parameters:**
- `target_min_side`: Minimum dimension for upscaling (default: 640px)
- `denoise_h`: Denoising strength for luminance (default: 10)
- `unsharp_amount`: Sharpening intensity (default: 0.7)
- `unsharp_sigma`: Gaussian blur sigma for unsharp mask (default: 1.0)

**Example:**
```python
from quality_enhancement import QualityEnhancer, QualityEnhancementConfig

config = QualityEnhancementConfig(
    target_min_side=640,
    unsharp_amount=0.8
)
enhancer = QualityEnhancer(config)
enhanced_img = enhancer(low_quality_img)
```

### Motion Deblurring (`motion_deblur.py`)

Removes motion blur using an optimized two-stage Richardson-Lucy deconvolution with parallel PSF search:

1. **Blur Detection**: Uses variance of Laplacian on center crop to detect blurry images
2. **Preview PSF Search**: Runs cheap deconvolution on downscaled grayscale preview to find best kernel (parallelized)
3. **Full-Resolution Deconvolution**: Applies single high-iteration Richardson-Lucy with chosen PSF on full color image
4. **Sharpening**: Final unsharp masking for edge enhancement

**Performance Features:**
- **Two-stage optimization**: Fast preview search eliminates expensive full-res trials
- **Parallel PSF evaluation**: ThreadPool or ProcessPool for concurrent kernel testing
- **Precomputed kernels**: PSFs generated once during initialization
- **Early stopping**: Optional convergence detection to reduce unnecessary iterations

**Key Parameters:**
- `lengths`: PSF kernel lengths to search (default: [9, 15, 21, 31])
- `angles`: PSF angles in degrees (default: [0, 30, 60, 90, 120, 150])
- `iterations`: Richardson-Lucy iterations for full-res (default: 15)
- `blur_threshold`: Laplacian variance threshold (default: 200.0)
- `pre_denoise`: Denoise before deblurring (default: True)
- `preview_scale`: Downscale factor for PSF search (default: 0.5)
- `preview_iterations`: RL iterations for preview search (default: 5)
- `sharpness_crop`: Center crop size for sharpness metric (default: 256)
- `use_thread_pool`: Enable parallel PSF search with threads (default: True)
- `use_process_pool`: Use processes instead of threads (default: False)
- `max_workers`: Number of parallel workers (default: None = auto)

**Example:**
```python
from motion_deblur import MotionDeblur, MotionDeblurConfig

# High-quality deblurring with parallelization
config = MotionDeblurConfig(
    lengths=(9, 15, 21, 31),
    angles=tuple(float(i) for i in range(0, 180, 15)),  # 12 angles
    iterations=100,
    use_thread_pool=True,
    max_workers=16
)
deblurrer = MotionDeblur(config)
deblurred_img = deblurrer(blurry_img)
```

**Batch Processing:**
```python
from motion_deblur import deblur_images_threadpool

# Process multiple images concurrently
images = [cv2.imread(f"img{i}.jpg") for i in range(10)]
results = deblur_images_threadpool(images, config, max_workers=4)
```

### Light Correction (`light_correction.py`)

Corrects lighting and color issues through three techniques:

1. **White Balance**: Gray-world assumption to remove color casts
2. **CLAHE**: Contrast Limited Adaptive Histogram Equalization on L channel
3. **Adaptive Gamma**: Automatically adjusts brightness to target mean

**Key Parameters:**
- `clahe_clip_limit`: CLAHE clipping limit (default: 2.0)
- `clahe_tile_grid_size`: CLAHE grid size (default: (8, 8))
- `gamma_target_mean`: Target brightness level (default: 0.5)
- `gamma_min` / `gamma_max`: Gamma correction bounds (default: 0.7 - 1.5)

**Example:**
```python
from light_correction import LightCorrection, LightCorrectionConfig

config = LightCorrectionConfig(
    clahe_clip_limit=2.5,
    gamma_target_mean=0.55
)
corrector = LightCorrection(config)
corrected_img = corrector(dark_img)
```

## Installation

### Requirements

```bash
pip install opencv-python numpy matplotlib
```

For Jupyter notebook support:
```bash
pip install jupyter notebook
```

### Dependencies

- Python 3.10+
- OpenCV (cv2) >= 4.0
- NumPy >= 1.20
- Matplotlib >= 3.0 (for visualization only)

## Usage

### Demo Notebook

See `traditional_IP_demo.ipynb` for interactive examples with visualizations:
- Light correction on underexposed images
- Motion deblurring on blurred photos
- Quality enhancement on downsampled images

Run the notebook:
```bash
jupyter notebook traditional_IP_demo.ipynb
```

## Algorithm Details

### Richardson-Lucy Deconvolution

The motion deblurring module implements the Richardson-Lucy algorithm, an iterative deconvolution method that maximizes likelihood under Poisson noise assumptions:

```
latent_{n+1} = latent_n * [(observed / (latent_n * PSF)) * PSF_mirror]
```

**Two-Stage Optimization Strategy:**

To efficiently search across many PSF candidates, the implementation uses a two-stage approach:

1. **Preview Stage**: Downscales the image (default 0.5x) and converts to grayscale, then runs low-iteration Richardson-Lucy (5 iterations) on each PSF candidate in parallel using ThreadPool/ProcessPool. Sharpness is measured on a center crop to reduce computation.

2. **Full-Resolution Stage**: Once the best PSF is identified from the preview, runs a single high-iteration Richardson-Lucy deconvolution (15-100+ iterations) on the full-resolution color image.

This reduces computational cost from `O(N_psf * N_iter * resolution)` to approximately `O(N_psf * 5 * 0.25 * resolution + N_iter * resolution)`, enabling searches over hundreds of PSF candidates in reasonable time.

**Early Stopping**: The Richardson-Lucy implementation supports optional convergence detection, checking every N iterations whether the change in latent image falls below a tolerance threshold.

### CLAHE (Contrast Limited Adaptive Histogram Equalization)

Enhances local contrast by applying histogram equalization on small tiles while limiting amplification to avoid noise over-enhancement. Applied in LAB color space to preserve color information.

### Unsharp Masking

Classical sharpening technique:
```
sharpened = original + amount * (original - gaussian_blur(original))
```

## Configuration Tips

**For heavily blurred images:**
- Increase `iterations` to 50-100 for stronger deblurring
- Expand `lengths` search range to include larger kernels (e.g., 41, 51)
- Increase `angles` density: `tuple(float(i) for i in range(0, 180, 10))` for 18 angles
- Enable `pre_denoise` to reduce noise amplification

**For performance optimization:**
- Enable `use_thread_pool=True` for parallel PSF search (works well with OpenCV's GIL release)
- Set `use_process_pool=True` for CPU-heavy workloads, though it has higher overhead
- Adjust `max_workers` based on CPU cores (typically 4-16)
- Reduce `preview_scale` to 0.25 for faster preview search on large images
- Decrease `preview_iterations` to 3 if PSF search is too slow
- Use `deblur_images_threadpool()` for batch processing multiple images

**For very dark/light images:**
- Adjust `gamma_target_mean` (lower for dark, higher for bright)
- Increase `clahe_clip_limit` for stronger local contrast

**For noisy images:**
- Increase `denoise_h` in quality enhancement
- Enable `pre_denoise` in motion deblurring
- Lower `unsharp_amount` to avoid amplifying noise
- Consider reducing `iterations` as high iterations can amplify noise

## Limitations

- Motion deblurring assumes linear motion blur (not suitable for complex blur patterns, camera shake, or defocus blur)
- Richardson-Lucy can amplify noise in heavily degraded images, especially with high iteration counts
- Preview-based PSF search may occasionally miss the optimal kernel, trading accuracy for speed
- Processing time still scales with image size, iteration count, and number of PSF candidates (despite optimizations)
- ThreadPool performance depends on GIL release; ProcessPool has pickling overhead
- No learning-based methods; performance limited by classical techniques