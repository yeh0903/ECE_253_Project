from pathlib import Path
from io import BytesIO
import csv
from PIL import Image, ImageFilter

# Paths 
INPUT_DIR  = Path(r"E:\2025 fall\Fundamentals of Digital Image Processing\compress\Raw")
OUTPUT_DIR = Path(r"E:\2025 fall\Fundamentals of Digital Image Processing\compress\Compress")
LOG_CSV    = OUTPUT_DIR / "compress_log.csv"

# Compression config (default: JPEG compression only, no resize)
QUALITY = 18           # Strength of JPEG compression: 15–30 gives stronger artifacts
SUBSAMPLING = 2        # 4:2:0
PROGRESSIVE = False
OPTIMIZE = True
KEEP_EXIF = True

DOWNSCALE_LONG = 320   # None means no downscale; set 256/320 etc. to increase distortion, then optionally upscale back
UPSCALE_BACK = True
GAUSS_BLUR_SIGMA = 0.8 # Slight Gaussian blur (e.g., 0.8). Set 0 to disable.
REENCODE_PASSES = 2    # Number of repeated JPEG re-encodes (2 or 3 recommended)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def save_jpeg_bytes(img: Image.Image, quality: int) -> bytes:
    """Safely save as JPEG and return raw bytes; skip EXIF arg if missing to avoid NoneType errors."""
    buf = BytesIO()
    exif_bytes = img.info.get("exif") if KEEP_EXIF else None
    kwargs = dict(
        format="JPEG",
        quality=int(quality),
        subsampling=SUBSAMPLING,
        progressive=PROGRESSIVE,
        optimize=OPTIMIZE,
    )
    if exif_bytes:  # Only pass EXIF if it exists
        kwargs["exif"] = exif_bytes
    img.save(buf, **kwargs)
    return buf.getvalue()


def downscale_then_upscale(im: Image.Image, long_side):
    if long_side is None:
        return im
    w, h = im.size
    if max(w, h) <= long_side:
        return im
    if w >= h:
        new_w = long_side
        new_h = int(h * long_side / w)
    else:
        new_h = long_side
        new_w = int(w * long_side / h)
    im_small = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
    if UPSCALE_BACK:
        return im_small.resize((w, h), Image.Resampling.BICUBIC)
    return im_small


def ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img.convert("RGBA"), mask=img.split()[-1])
        return bg
    return img.convert("RGB")


def process_one(src: Path, out_dir: Path) -> Path:
    im = Image.open(src)
    im.load()
    im = ensure_rgb(im)

    if GAUSS_BLUR_SIGMA and GAUSS_BLUR_SIGMA > 0:
        im = im.filter(ImageFilter.GaussianBlur(radius=GAUSS_BLUR_SIGMA))

    im = downscale_then_upscale(im, DOWNSCALE_LONG)

    # Multiple JPEG re-encodes
    tmp = im
    for _ in range(REENCODE_PASSES):
        b = save_jpeg_bytes(tmp, QUALITY)
        tmp = Image.open(BytesIO(b)).convert("RGB")

    # Output (keep original stem, force .jpg)
    stem = src.stem
    out_path = out_dir / f"{stem}.jpg"
    out_path.write_bytes(save_jpeg_bytes(tmp, QUALITY))
    return out_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = [p for p in INPUT_DIR.rglob("*") if p.suffix.lower() in IMG_EXTS]
    total = len(files)

    with LOG_CSV.open("w", newline="", encoding="utf-8") as fcsv:
        wr = csv.writer(fcsv)
        wr.writerow(["src", "out", "quality", "downscale_long", "passes", "bytes"])
        for i, p in enumerate(files, 1):
            try:
                outp = process_one(p, OUTPUT_DIR)
                wr.writerow(
                    [str(p), str(outp), QUALITY, DOWNSCALE_LONG, REENCODE_PASSES, outp.stat().st_size]
                )
                print(f"[{i}/{total}] -> {outp.name}")
            except Exception as e:
                print(f"[ERROR] {p} ({e})")

    print("\n[DONE] Output:", OUTPUT_DIR)
    print("[LOG]      ", LOG_CSV)


if __name__ == "__main__":
    main()
