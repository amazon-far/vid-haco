"""
Where a point of an original ARCTIC image lands in ARCTIC's cropped image, for the allocentric views.

For cameras 1-8, ARCTIC's `cropped_images` are a square window of side 1.5 * 200 * scale pixels, centred
on the per-frame (cx, cy, scale) box shipped in the split's data_dict['bbox'], resized to 1000 x 1000.
"""
import numpy as np

CROP_DIM = 1000


def to_crop_coords(points, box, dim=CROP_DIM):
    """(N, 2) original-image pixels -> pixels in the dim x dim crop of `box` = (cx, cy, scale)."""
    cx, cy, scale = (float(v) for v in box)
    side = 300.0 * scale
    corner = np.array([cx - 0.5 * side, cy - 0.5 * side])
    return (np.asarray(points, np.float64) - corner) * (dim / side)
