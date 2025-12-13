"""
Offline FBCNN JPEG artifact removal (no internet, no torch.hub).
Folder structure (as you already have):
  ML_restore/
    ├─ ML_restore.py             (this file)
    ├─ FBCNN-main/               (https://github.com/jiaxi-jiang/FBCNN)
    ├─ fbcnn_color.pth           (pretrained weights)
    └─ phone_restored/           (auto-created outputs)
Input compressed images:
  E:\2025 fall\Fundamentals of Digital Image Processing\compress\phone_compressed
"""

from pathlib import Path
import re
import sys
import inspect
import numpy as np
from PIL import Image
import torch

# Paths 
ROOT   = Path(r"E:\2025 fall\Fundamentals of Digital Image Processing\ML_restore")
IN_DIR = Path(r"E:\2025 fall\Fundamentals of Digital Image Processing\Low_Resolution\images")
OUT_DIR = ROOT / "Low-Resloution-FBCNN"
FBCNN_DIR = ROOT / "FBCNN-main"
WEIGHTS   = ROOT / "fbcnn_color.pth"

OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Import FBCNN locally 
sys.path.insert(0, str(FBCNN_DIR.resolve()))
try:
    from models.network_fbcnn import FBCNN as FBCNN
except Exception as e:
    raise RuntimeError(
        f"Failed to import FBCNN. Please check if it exists at: {FBCNN_DIR}\\models\\network_fbcnn.py\n"
        f"Underlying error: {e}"
    )

# I/O 
def pick_image_tensor(y):
    "Some FBCNN implementations return a tuple; pick the 4D image tensor from it."
    if isinstance(y, (tuple, list)):
        for t in y:
            if torch.is_tensor(t) and t.dim() == 4:   # NCHW
                return t
        raise TypeError("Model returned a tuple/list but no 4D tensor was found.")
    if not torch.is_tensor(y):
        raise TypeError(f"Unexpected output type: {type(y)}")
    return y


def load_rgb(path: Path) -> torch.Tensor:
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)  # 1x3xHxW


def save_rgb(t, path: Path, q=95):
    t = pick_image_tensor(t)
    t = t.clamp(0, 1).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    Image.fromarray(np.round(t * 255.0).astype(np.uint8), mode="RGB").save(
        path, quality=q, subsampling=0
    )


# Utils 
def guess_q_from_name(name: str, default_q=50) -> int:
    m = re.search(r"[Qq](\d{1,3})", name)
    if m:
        q = int(m.group(1))
        return int(min(95, max(10, q)))
    return default_q


def load_state_into(model: torch.nn.Module, ckpt_path: Path):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Weight file not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state_dict", None) or ckpt.get("state_dict", None) or ckpt
    else:
        state = ckpt
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        state2 = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state2, strict=False)


def infer_input_channels_from_state(ckpt_path: Path, fallback=3) -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", None) or ckpt.get("state_dict", None) or ckpt
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and v.ndim == 4 and "m_head" in k and "weight" in k:
            return int(v.shape[1])
    # Fallback: try to find any first convolution layer
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and v.ndim == 4 and "weight" in k:
            return int(v.shape[1])
    return fallback


def build_fbcnn(in_nc: int):
    "Support different FBCNN constructor signatures: some use nf/nb, others use nc(list)/nb(list or int)."
    try:
        sig = inspect.signature(FBCNN.__init__)
        names = set(sig.parameters.keys())
    except Exception:
        names = set()

    if "nc" in names:
        # Typical upstream version: nc is a list
        # Try nb as int first; if it fails, try list
        try:
            return FBCNN(in_nc=in_nc, out_nc=3, nc=[64, 128, 256, 512], nb=4)
        except TypeError:
            try:
                return FBCNN(in_nc=in_nc, out_nc=3, nc=[64, 128, 256, 512], nb=[1, 1, 1, 1])
            except TypeError:
                # Some implementations use same width for all layers
                try:
                    return FBCNN(in_nc=in_nc, out_nc=3, nc=[64, 64, 64, 64], nb=4)
                except Exception:
                    return FBCNN(in_nc=in_nc, out_nc=3, nc=[64, 64, 64, 64], nb=[1, 1, 1, 1])
    else:
        # KAIR-style variant: nf/nb
        try:
            return FBCNN(in_nc=in_nc, out_nc=3, nf=64, nb=8)
        except TypeError:
            return FBCNN(in_nc=in_nc, out_nc=3, nf=64)


# Main 
@torch.no_grad()
def run():
    files = [p for p in IN_DIR.rglob("*") if p.suffix.lower() in IMG_EXT]
    print(f"Device: {DEVICE} | Images: {len(files)} | Out: {OUT_DIR}")

    # Typically 4 (RGB + qmap)
    in_nc = infer_input_channels_from_state(WEIGHTS, fallback=3)
    net = build_fbcnn(in_nc)
    load_state_into(net, WEIGHTS)
    net = net.to(DEVICE).eval()

    for p in files:
        try:
            x = load_rgb(p)                     # 1x3xHxW
            H, W = x.shape[-2:]
            if in_nc == 4:
                q = guess_q_from_name(p.name, default_q=50)
                qmap = torch.full((1, 1, H, W), float(q) / 100.0, device=DEVICE)
                # Prefer forward(x, qmap); if that fails, concat along channel dim
                try:
                    y = net(x, qmap)
                except Exception:
                    y = net(torch.cat([x, qmap], dim=1))
            else:
                # Blind model
                try:
                    y = net(x)
                except Exception:
                    qmap = torch.full((1, 1, H, W), 0.5, device=DEVICE)
                    try:
                        y = net(x, qmap)
                    except Exception:
                        y = net(torch.cat([x, qmap], dim=1))

            out_path = OUT_DIR / f"{p.stem}.jpg"
            save_rgb(y, out_path, q=95)
            print("saved:", out_path)
        except Exception as e:
            print("[ERROR]", p, "->", e)


if __name__ == "__main__":
    run()
