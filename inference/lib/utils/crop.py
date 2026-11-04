"""
Hand crops: grow a detector box around the hand and map it onto the model's square input.

Stage A turns each detection into a box twice the hand's size (so the crop also shows what the hand is
touching) and smooths it over time; Stage B warps every frame of a clip with the affine of its target
frame's box. The box is stretched to fill the crop — each side is scaled independently, there is no
aspect-ratio padding.
"""
import cv2
import numpy as np

BOX_EXPAND = 2.0


def expand_box(xyxy, ratio=BOX_EXPAND):
    """Detector box (x1, y1, x2, y2) -> [x, y, w, h], scaled by `ratio` about its centre."""
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    w, h = ratio * (x2 - x1), ratio * (y2 - y1)
    return [0.5 * (x1 + x2) - 0.5 * w, 0.5 * (y1 + y2) - 0.5 * h, w, h]


def crop_transform(box, out_w, out_h):
    """2x3 affine taking box [x, y, w, h] onto an out_w x out_h image, centre onto centre."""
    x, y, w, h = (float(v) for v in box)
    sx, sy = out_w / w, out_h / h
    return np.array([[sx, 0.0, 0.5 * out_w - sx * (x + 0.5 * w)],
                     [0.0, sy, 0.5 * out_h - sy * (y + 0.5 * h)]], np.float32)


def warp(img, M, out_w, out_h, fill=0):
    """Apply a crop affine (bilinear); pixels outside the source image get `fill`."""
    return cv2.warpAffine(img, M, (int(out_w), int(out_h)), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(fill, fill, fill))
