import io

import cv2
import numpy as np
import pytest
from PIL import Image

from utils.digit_detection import extract_digits
from utils.preprocessing import (SegmentationParams, foreground_mask, is_light_background, load_image,
                                 normalize_digit, to_grayscale)


def make_image(text="201", dark_on_light=True, size=(420, 160)):
    img = np.full((size[1], size[0]), 255 if dark_on_light else 0, np.uint8)
    cv2.putText(img, text, (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 4, 0 if dark_on_light else 255, 14)
    return img


def png_bytes(arr, mode=None):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_load_rgb_rgba_gray():
    g = make_image()
    rgb = np.stack([g] * 3, -1)
    rgba = np.concatenate([rgb, np.full(g.shape + (1,), 255, np.uint8)], -1)
    for arr in (g, rgb, rgba):
        out = to_grayscale(load_image(png_bytes(arr)))
        assert out.ndim == 2 and out.dtype == np.uint8 and out.shape == g.shape


def test_transparent_png_flattened_on_white():
    rgba = np.zeros((20, 20, 4), np.uint8)  # fully transparent
    out = to_grayscale(load_image(png_bytes(rgba)))
    assert out.min() == 255


def test_exif_orientation_applied():
    img = Image.fromarray(np.tile(np.arange(40, dtype=np.uint8), (20, 1)))  # 40 wide x 20 tall
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    assert load_image(buf.getvalue()).size == (20, 40)


def test_background_polarity_detection():
    assert is_light_background(make_image(dark_on_light=True))
    assert not is_light_background(make_image(dark_on_light=False))


@pytest.mark.parametrize("dark", [True, False])
def test_foreground_is_minority_and_white(dark):
    mask, info = foreground_mask(make_image(dark_on_light=dark))
    assert info["light_background"] == dark
    assert 0 < (mask > 0).mean() < 0.4  # digits, not the paper, are foreground


@pytest.mark.parametrize("method", ["otsu", "adaptive", "manual"])
def test_all_threshold_methods_find_digits(method):
    ex = extract_digits(make_image(), SegmentationParams(method=method, threshold=100))
    assert len(ex.digits) == 3


def test_detects_and_orders_digits_left_to_right():
    ex = extract_digits(make_image("201"))
    assert len(ex.boxes) == 3
    xs = [b.cx for b in ex.boxes]
    assert xs == sorted(xs)
    ex2 = extract_digits(make_image("12345", size=(700, 160)))
    assert len(ex2.boxes) == 5


def test_inverted_image_gives_same_result():
    a = extract_digits(make_image("786"))
    b = extract_digits(make_image("786", dark_on_light=False))
    assert len(a.digits) == len(b.digits) == 3


def test_normalised_digit_format():
    ex = extract_digits(make_image("201"))
    for d in ex.digits:
        assert d.shape == (28, 28) and d.dtype == np.float32
        assert 0.0 <= d.min() and d.max() == pytest.approx(1.0)
        assert d[:1].sum() == 0 and d[-1:].sum() == 0 and d[:, :1].sum() == 0 and d[:, -1:].sum() == 0
    plain = extract_digits(make_image("201"), SegmentationParams(center_by_mass=False))
    for d in plain.digits:  # without COM shift the digit sits inside the centred 20x20 box
        assert d[:4].sum() == 0 and d[-4:].sum() == 0 and d[:, :4].sum() == 0 and d[:, -4:].sum() == 0


def test_aspect_ratio_preserved():
    wide = np.full((10, 60), 255, np.uint8)  # a bar 6:1
    out = normalize_digit(wide, normalize_stroke=False, center_by_mass=False)
    ys, xs = np.nonzero(out > 0.5)
    assert (xs.max() - xs.min() + 1) == 20 and (ys.max() - ys.min() + 1) <= 5


def test_empty_input_gives_blank_digit():
    assert normalize_digit(np.zeros((5, 5), np.uint8)).sum() == 0
    assert extract_digits(np.full((50, 50), 200, np.uint8)).digits == []


def test_broken_digit_pieces_are_merged():
    img = np.full((120, 80), 255, np.uint8)
    cv2.rectangle(img, (20, 10), (60, 25), 0, -1)   # top bar
    cv2.rectangle(img, (20, 35), (60, 100), 0, -1)  # body, separate component right underneath
    assert len(extract_digits(img, SegmentationParams(merge_components=True)).boxes) == 1
    assert len(extract_digits(img, SegmentationParams(merge_components=False)).boxes) == 2
