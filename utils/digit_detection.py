"""Digit localisation: connected components -> merged, ordered digit boxes -> 28x28 digits."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from utils.preprocessing import SegmentationParams, foreground_mask, normalize_digit, tight_crop


@dataclass
class DigitBox:
    x: int
    y: int
    w: int
    h: int
    labels: list[int] = field(default_factory=list)  # connected-component labels that make up this digit

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


@dataclass
class Extraction:
    mask: np.ndarray
    boxes: list[DigitBox]
    crops: list[np.ndarray]      # binary crops (255 = digit), one per box
    digits: list[np.ndarray]     # float32 [28, 28] each
    info: dict


def _union(a: DigitBox, b: DigitBox) -> DigitBox:
    x, y = min(a.x, b.x), min(a.y, b.y)
    return DigitBox(x, y, max(a.x2, b.x2) - x, max(a.y2, b.y2) - y, a.labels + b.labels)


def _should_merge(a: DigitBox, b: DigitBox) -> bool:
    """Merge pieces stacked on top of each other (e.g. the open top bar of a 5, or the two parts of a 4)."""
    overlap = min(a.x2, b.x2) - max(a.x, b.x)
    if overlap <= 0:
        return False
    if overlap / min(a.w, b.w) < 0.5:
        return False
    vgap = max(a.y, b.y) - min(a.y2, b.y2)  # negative when they overlap vertically
    return vgap < 0.5 * max(a.h, b.h)


def order_boxes(boxes: list[DigitBox]) -> list[DigitBox]:
    """Reading order: rows top-to-bottom, left-to-right within each row."""
    if not boxes:
        return []
    med_h = float(np.median([b.h for b in boxes]))
    rows: list[list[DigitBox]] = []
    for b in sorted(boxes, key=lambda b: b.cy):
        for row in rows:
            if abs(np.mean([r.cy for r in row]) - b.cy) < 0.6 * med_h:
                row.append(b)
                break
        else:
            rows.append([b])
    rows.sort(key=lambda r: np.mean([b.cy for b in r]))
    return [b for row in rows for b in sorted(row, key=lambda b: b.cx)]


def detect_boxes(mask: np.ndarray, params: SegmentationParams | None = None) -> tuple[list[DigitBox], np.ndarray]:
    """Find digit boxes in a binary mask. Returns (ordered boxes, label image)."""
    p = params or SegmentationParams()
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    comps = [
        DigitBox(int(stats[i, 0]), int(stats[i, 1]), int(stats[i, 2]), int(stats[i, 3]), [i])
        for i in range(1, n)
        if stats[i, cv2.CC_STAT_AREA] >= 6
    ]
    if not comps:
        return [], labels

    if p.merge_components:
        changed = True
        while changed:
            changed = False
            for i in range(len(comps)):
                for j in range(i + 1, len(comps)):
                    if _should_merge(comps[i], comps[j]):
                        comps[i] = _union(comps[i], comps[j])
                        del comps[j]
                        changed = True
                        break
                if changed:
                    break

    max_h = max(c.h for c in comps)
    comps = [c for c in comps if c.h >= p.min_height_frac * max_h]  # speckles / punctuation
    return order_boxes(comps), labels


def box_crop(mask: np.ndarray, labels: np.ndarray, box: DigitBox) -> np.ndarray:
    """Binary crop containing only the pixels of this digit's components (not its neighbours')."""
    sub = labels[box.y:box.y2, box.x:box.x2]
    return (np.isin(sub, box.labels) * 255).astype(np.uint8)


def extract_digits(gray: np.ndarray, params: SegmentationParams | None = None) -> Extraction:
    """Full automatic pipeline: gray image -> ordered normalised 28x28 digits."""
    p = params or SegmentationParams()
    mask, info = foreground_mask(gray, p)
    boxes, labels = detect_boxes(mask, p)
    crops = [box_crop(mask, labels, b) for b in boxes]
    digits = [normalize_digit(c, p.normalize_stroke, p.center_by_mass) for c in crops]
    return Extraction(mask, boxes, crops, digits, info)


def digit_from_region(gray: np.ndarray, region: tuple[int, int, int, int], params: SegmentationParams | None = None):
    """Manual correction: treat everything inside region=(x0, y0, x1, y1) as one digit.

    Polarity and threshold are decided on the *whole* image so the result is consistent with
    the automatic path. Returns (digit28 or None, binary crop or None).
    """
    p = params or SegmentationParams()
    mask, _ = foreground_mask(gray, p)
    x0, y0, x1, y1 = region
    sub = mask[max(0, y0):y1, max(0, x0):x1]
    crop = tight_crop(sub)
    if crop is None:
        return None, None
    return normalize_digit(crop, p.normalize_stroke, p.center_by_mass), crop


def single_digit(gray: np.ndarray, params: SegmentationParams | None = None):
    """Whole image is one digit (used for the drawing canvas)."""
    return digit_from_region(gray, (0, 0, gray.shape[1], gray.shape[0]), params)


def draw_boxes(gray: np.ndarray, boxes: list[DigitBox], selected: set[int] | None = None) -> Image.Image:
    """RGB preview of the image with numbered boxes (green = used, grey = excluded)."""
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    th = max(2, min(gray.shape) // 200)
    for i, b in enumerate(boxes):
        on = selected is None or i in selected
        color = (0, 158, 115) if on else (150, 150, 150)
        cv2.rectangle(rgb, (b.x, b.y), (b.x2, b.y2), color, th)
        cv2.putText(rgb, str(i + 1), (b.x, max(12, b.y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.5, min(gray.shape) / 400), color, th)
    return Image.fromarray(rgb)
