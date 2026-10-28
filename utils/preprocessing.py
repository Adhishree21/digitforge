"""Image loading, foreground segmentation and MNIST-style 28x28 normalisation.

Convention used everywhere (matches MNIST and the training pipeline):
white digit (1.0) on black background (0.0), float32 in [0, 1], shape [28, 28].
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.morphology import skeletonize

MAX_SIDE = 1600  # larger uploads are down-scaled for speed


# ----------------------------------------------------------------------------- loading
def load_image(source) -> Image.Image:
    """Open bytes / file-like / path, fix EXIF orientation, flatten transparency onto white."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    img = Image.open(source)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba).convert("RGB")
    elif img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img


def to_grayscale(img: Image.Image) -> np.ndarray:
    """PIL image -> uint8 [H, W] grayscale, down-scaled if very large."""
    gray = np.array(img.convert("L"), dtype=np.uint8)
    h, w = gray.shape
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        gray = cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return gray


# ----------------------------------------------------------------------------- polarity
def _otsu(gray: np.ndarray) -> float:
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t)


def is_light_background(gray: np.ndarray) -> bool:
    """True if the background is brighter than the ink (dark digit on light paper).

    The border strip is almost always background, so its median is compared with the Otsu
    threshold of the whole image. If the image is nearly flat, fall back to the global median.
    """
    h, w = gray.shape
    b = max(2, int(0.03 * min(h, w)))
    border = np.concatenate([gray[:b].ravel(), gray[-b:].ravel(), gray[:, :b].ravel(), gray[:, -b:].ravel()])
    if int(gray.max()) - int(gray.min()) < 20:
        return float(np.median(gray)) > 127
    return float(np.median(border)) > _otsu(gray)


# ----------------------------------------------------------------------------- segmentation
@dataclass
class SegmentationParams:
    invert: str = "auto"           # "auto" | "dark_on_light" | "light_on_dark"
    method: str = "otsu"           # "otsu" | "adaptive" | "manual"
    threshold: int = 128           # used when method == "manual" (applied to the "ink" image)
    merge_components: bool = True
    min_height_frac: float = 0.2   # components shorter than this fraction of the tallest are noise
    normalize_stroke: bool = True
    center_by_mass: bool = True


def make_ink_image(gray: np.ndarray, invert: str = "auto") -> tuple[np.ndarray, bool]:
    """Return an image where ink is bright, and whether the paper was detected as light."""
    if invert == "auto":
        light = is_light_background(gray)
    else:
        light = invert == "dark_on_light"
    ink = 255 - gray if light else gray.copy()
    return ink, light


def foreground_mask(gray: np.ndarray, params: SegmentationParams | None = None) -> tuple[np.ndarray, dict]:
    """Binary mask (255 = digit) plus info dict {light_background, threshold}."""
    p = params or SegmentationParams()
    ink, light = make_ink_image(gray, p.invert)
    ink = cv2.GaussianBlur(ink, (3, 3), 0)
    info = {"light_background": light, "threshold": None}
    if int(ink.max()) - int(ink.min()) < 25:  # no contrast -> nothing to extract
        return np.zeros_like(gray), info

    if p.method == "manual":
        _, mask = cv2.threshold(ink, int(p.threshold), 255, cv2.THRESH_BINARY)
        info["threshold"] = float(p.threshold)
    elif p.method == "adaptive":
        block = max(11, (min(gray.shape) // 8) | 1)
        mask = cv2.adaptiveThreshold(ink, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, -10)
        mask[ink < 40] = 0  # adaptive thresholding amplifies noise on flat background
        info["threshold"] = None
    else:
        t, mask = cv2.threshold(ink, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        info["threshold"] = float(t)
    return mask.astype(np.uint8), info


# ----------------------------------------------------------------------------- normalisation
def _estimate_stroke_width(mask01: np.ndarray) -> float:
    """Average stroke width in pixels = foreground area / skeleton length."""
    area = float(mask01.sum())
    skel = float(skeletonize(mask01.astype(bool)).sum())
    return area / skel if skel > 0 else 1.0


def tight_crop(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def normalize_digit(
    mask_crop: np.ndarray,
    normalize_stroke: bool = True,
    center_by_mass: bool = True,
    box: int = 20,
    canvas: int = 28,
    target_stroke: float = 2.2,
) -> np.ndarray:
    """Binary digit crop -> MNIST-style float32 [28, 28].

    1. crop margins   2. keep aspect ratio and fit into box x box   3. centre on the canvas
    4. scale to [0, 1]. MNIST strokes are ~2 px wide at this scale, so thin pen strokes are
    optionally dilated *before* resizing to reach that width (thick strokes are left alone).
    """
    crop = tight_crop(mask_crop)
    if crop is None:
        return np.zeros((canvas, canvas), np.float32)
    m = (crop > 0).astype(np.uint8)
    h, w = m.shape
    scale = box / max(h, w)

    if normalize_stroke:
        stroke = _estimate_stroke_width(m)
        deficit = target_stroke / scale - stroke
        if deficit > 0.5:
            r = max(1, int(round(deficit / 2)))
            pad = r + 1
            m = np.pad(m, pad)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            m = cv2.dilate(m, k)
            m = tight_crop(m)
            h, w = m.shape
            scale = box / max(h, w)

    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    small = cv2.resize(m.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_AREA)
    out = np.zeros((canvas, canvas), np.float32)
    x0, y0 = (canvas - new_w) // 2, (canvas - new_h) // 2
    out[y0:y0 + new_h, x0:x0 + new_w] = small

    if center_by_mass and out.sum() > 0:
        ys, xs = np.indices(out.shape)
        cy, cx = (ys * out).sum() / out.sum(), (xs * out).sum() / out.sum()
        dx, dy = int(round((canvas - 1) / 2 - cx)), int(round((canvas - 1) / 2 - cy))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        out = cv2.warpAffine(out, M, (canvas, canvas), flags=cv2.INTER_NEAREST, borderValue=0)

    mx = out.max()
    if mx > 0:
        out = out / mx
    return np.clip(out, 0.0, 1.0).astype(np.float32)
