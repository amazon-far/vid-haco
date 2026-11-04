"""Image I/O, hand-crop geometry and crop augmentation shared by every Vid-HACO data reader.

Conventions
  image   RGB float32 array (H, W, 3) with values in [0, 255]
  box     hand box (x, y, w, h) in pixels of the image it refers to
  crop    (out_h, out_w) patch cut from an image with a 2x3 affine M (image pixels -> crop pixels),
          resampled bilinearly for RGB, nearest-neighbour for masks, zero outside the image

The training-time augmentation reproduces the distribution of the crops HACO (Jung & Lee,
NeurIPS 2025) trains on, so the paper's training mix is unchanged; the code here is independent.
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Every dataset lives under ./data/<NAME>/, relative to the working directory.
DATA_ROOT = 'data'

# Network input size (height, width). 256 x 256 is what the readers have always produced (the
# V-JEPA 2 ViT-H/256 model); the V-JEPA 2.1 ViT-L/384 model is trained on (384, 384). Every reader
# takes an `input_size` argument; None means this value, looked up when the reader is built.
DEFAULT_INPUT_SIZE = (256, 256)

# ImageNet statistics == the V-JEPA 2 video-processor defaults.
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)


def input_hw(input_size=None):
    """(height, width) ints from an int, a pair, or None (-> DEFAULT_INPUT_SIZE)."""
    size = DEFAULT_INPUT_SIZE if input_size is None else input_size
    if np.isscalar(size):
        size = (size, size)
    return int(size[0]), int(size[1])


def read_rgb(path, mirror=False):
    """Decode an image file to RGB float32 (H, W, 3), optionally mirrored left-right. EXIF
    orientation is ignored, because the boxes refer to the stored pixel grid. Raises IOError when
    the file cannot be decoded."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if bgr is None:
        raise IOError(f'cannot read image: {path}')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    return cv2.flip(rgb, 1) if mirror else rgb


def load_npz_key(path, key, allow_pickle=False):
    with np.load(path, allow_pickle=allow_pickle) as z:
        return z[key]


def load_contact(path):
    """778-vertex contact vector stored as .npy -> float32."""
    return np.load(path).astype(np.float32)


# ------------------------------------------------------------------------------ augmentation
@dataclass
class CropAug:
    """One draw of crop augmentation; the default instance is the identity (test-time crop)."""
    scale: float = 1.0                       # crop window = box size * scale
    rot: float = 0.0                         # rotation of the window about its centre, degrees
    shift: Tuple[float, float] = (0.0, 0.0)  # window-centre offset, fraction of image (W, H)
    flip: bool = False                       # mirror the image (and the box centre) first
    gain: np.ndarray = field(default_factory=lambda: np.ones(3))   # per-channel colour gain
    # degradations, used only for single-image crops (crop_image(..., degrade=True)):
    low_res: float = 1.0                     # < 1: down-sample the crop by this factor, then up
    zoom: Optional[Tuple[int, int, int]] = None   # (x0, y0, side) square window, resized back up
    noise: Optional[np.ndarray] = None       # additive noise, (out_h, out_w, 3)
    blur: Optional[Tuple[int, str]] = None   # motion blur: (kernel size, 'h' | 'v' | 'd')


def sample_aug(input_size=None, rng=None, flip=False):
    """Draw training-crop augmentation once per sample (or clip); apply it to every frame.

    Distribution (the same as HACO's training crops):
      scale   1 + 0.25 * clip(N(0, 1), -1, 1)
      rot     30 * clip(N(0, 1), -2, 2) degrees with probability 0.6, else 0
      shift   0.1 * clip(N(0, 1), -1, 1) of the image width / height, per axis
      gain    U(0.8, 1.2) per RGB channel
      flip    `flip=False` (what every reader passes) never mirrors, True always, None with p 0.5
      low-res p 0.05, factor U(0.15, 0.5) (INTER_AREA down, nearest up)
      zoom    p 0.1, a random square window with side int(0.7 * shorter crop side), resized back
      noise   p 0.3, N(0, 0.02) per pixel (see crop_image for how it is applied)
      blur    p 0.15, linear motion blur, kernel 3 / 5 / 7, horizontal / vertical / diagonal

    `rng` is a numpy RandomState or Generator; the default is numpy's global RNG, which PyTorch
    reseeds in every DataLoader worker.
    """
    r = np.random if rng is None else rng
    out_h, out_w = input_hw(input_size)

    def bounded_normal(limit):
        return float(np.clip(r.standard_normal(), -limit, limit))

    aug = CropAug(scale=1.0 + 0.25 * bounded_normal(1.0))
    if r.random() < 0.6:
        aug.rot = 30.0 * bounded_normal(2.0)
    aug.gain = r.uniform(0.8, 1.2, 3)
    aug.flip = bool(r.random() < 0.5) if flip is None else bool(flip)
    aug.shift = (0.1 * bounded_normal(1.0), 0.1 * bounded_normal(1.0))
    if r.random() < 0.1:
        side = max(1, int(min(out_h, out_w) * 0.7))
        aug.zoom = (int(r.choice(out_w - side + 1)), int(r.choice(out_h - side + 1)), side)
    if r.random() < 0.3:
        aug.noise = r.normal(0.0, 0.02, (out_h, out_w, 3)).astype(np.float32)
    if r.random() < 0.15:
        aug.blur = (int(r.choice([3, 5, 7])), str(r.choice(['h', 'v', 'd'])))
    if r.random() < 0.05:
        aug.low_res = float(r.uniform(0.15, 0.5))
    return aug


# ------------------------------------------------------------------------------ geometry
def box_to_crop(box, input_size=None, aug=None, image_hw=None):
    """2x3 float32 affine taking image pixels to crop pixels for the hand box `box`.

    Without augmentation the box centre goes to the crop centre and the box fills the crop:
        cx = x + w/2, cy = y + h/2, sx = out_w / w, sy = out_h / h
        M  = [[sx, 0, out_w/2 - sx*cx],
              [0, sy, out_h/2 - sy*cy]]
    With `aug` the centre is first mirrored (cx -> W - 1 - cx, matching an image mirrored with
    cv2.flip) and shifted by aug.shift * (W, H) -- both need `image_hw` = (H, W) -- then the window
    is scaled by aug.scale and rotated by aug.rot degrees about the centre.
    Raises ValueError for a box without positive, finite width and height.
    """
    x, y, w, h = (float(v) for v in box)
    if not (0 < w < np.inf and 0 < h < np.inf):
        raise ValueError(f'degenerate hand box {(x, y, w, h)}')
    out_h, out_w = input_hw(input_size)
    cx, cy = x + w / 2.0, y + h / 2.0
    scale, rot = 1.0, 0.0
    if aug is not None:
        img_h, img_w = image_hw
        if aug.flip:
            cx = img_w - 1 - cx
        cx += aug.shift[0] * img_w
        cy += aug.shift[1] * img_h
        scale, rot = aug.scale, aug.rot
    sx, sy = out_w / (w * scale), out_h / (h * scale)
    if rot:
        # window axes are the image axes rotated by `rot`; undo that rotation, then stretch
        cos, sin = np.cos(np.deg2rad(rot)), np.sin(np.deg2rad(rot))
        a, b, c, d = sx * cos, sx * sin, -sy * sin, sy * cos
    else:
        a, b, c, d = sx, 0.0, 0.0, sy
    return np.array([[a, b, out_w / 2.0 - (a * cx + b * cy)],
                     [c, d, out_h / 2.0 - (c * cx + d * cy)]], dtype=np.float32)


def _motion_kernel(size, direction):
    k = np.zeros((size, size))
    if direction == 'h':
        k[size // 2, :] = 1.0
    elif direction == 'v':
        k[:, size // 2] = 1.0
    else:
        k[np.arange(size), np.arange(size)] = 1.0
    return k / size


def _zoom(img, window, interpolation):
    x0, y0, side = window
    rows, cols = img.shape[:2]
    return cv2.resize(img[y0:y0 + side, x0:x0 + side], (cols, rows), interpolation=interpolation)


def crop_image(img, M, input_size=None, aug=None, degrade=True):
    """Crop `img` (RGB, [0, 255]) with M and apply the photometric part of `aug`.

    Order: mirror -> warp -> low-res -> colour gain (clipped to [0, 255]) -> zoom -> noise -> blur.
    degrade=False keeps only mirror, warp and colour gain; that is what every frame of a video clip
    gets, while single-image samples also get the degradations drawn in `aug`.
    Returns a float array (out_h, out_w, 3).
    """
    aug = aug or CropAug()
    out_h, out_w = input_hw(input_size)
    src = cv2.flip(img, 1) if aug.flip else img
    crop = cv2.warpAffine(src, M, (out_w, out_h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if degrade and aug.low_res < 1.0:
        small = (max(1, int(out_w * aug.low_res)), max(1, int(out_h * aug.low_res)))
        crop = cv2.resize(cv2.resize(crop, small, interpolation=cv2.INTER_AREA), (out_w, out_h),
                          interpolation=cv2.INTER_NEAREST)
    crop = np.clip(crop * aug.gain, 0.0, 255.0)
    if not degrade:
        return crop
    if aug.zoom is not None:
        crop = _zoom(crop, aug.zoom, cv2.INTER_LINEAR)
    if aug.noise is not None:
        # Kept as in the paper's training runs: the noisy crop is clipped to [0, 1] although it is
        # in [0, 255] here, so these crops come out almost uniformly dark.
        crop = np.clip(crop + aug.noise, 0.0, 1.0)
    if aug.blur is not None:
        crop = cv2.filter2D(crop, -1, _motion_kernel(*aug.blur), borderType=cv2.BORDER_REFLECT)
    return crop


def crop_mask(mask, M, input_size=None, aug=None, degrade=True):
    """Binary mask of the source image -> {0, 1} float32 (out_h, out_w), pixel-aligned with
    crop_image(img, M, input_size, aug, degrade): same mirror, same affine (nearest neighbour) and,
    with degrade=True, the same zoom window. The mask must have the image's resolution."""
    aug = aug or CropAug()
    out_h, out_w = input_hw(input_size)
    m = np.asarray(mask, dtype=np.float32)
    if aug.flip:
        m = cv2.flip(m, 1)
    m = cv2.warpAffine(m, M, (out_w, out_h), flags=cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if degrade and aug.zoom is not None:
        m = _zoom(m, aug.zoom, cv2.INTER_NEAREST)
    return (m > 0.5).astype(np.float32)


_MEAN = np.asarray(RGB_MEAN).reshape(3, 1, 1)
_STD = np.asarray(RGB_STD).reshape(3, 1, 1)


def to_input(crop):
    """[0, 255] RGB crop (H, W, 3) -> ImageNet-normalized float32 tensor (3, H, W)."""
    chw = np.asarray(crop).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(np.ascontiguousarray((chw - _MEAN) / _STD, dtype=np.float32))


# ------------------------------------------------------------------------------ readers
@dataclass
class Target:
    """What is needed to crop one labelled sample (paths and labels; no pixels yet)."""
    image: str                          # labelled frame
    box: np.ndarray                     # (4,) float32 hand box (x, y, w, h); refers to the
                                        # mirrored frame when `mirror` is set
    contact: np.ndarray                 # (778,) float32 per-vertex contact label
    mirror: bool = False                # mirror the frame before cropping (left hand -> right)
    frame: Optional[int] = None         # frame number, for datasets with numbered frames
    mask: Optional[np.ndarray] = None   # hand+object foreground (bool) of the stored frame


class HandCropDataset(Dataset):
    """Base class of every reader: a fixed, ordered pool of labelled hand crops.

    A subclass fills `self.split` (one entry per sample, in pool order) and implements
    `sample_id(i)` and `target(i)`. Datasets whose frames are named <zero-padded number><ext> set
    `frame_digits` / `frame_ext`, which is how neighbouring clip frames are found.

    `data_split` only selects the crop of the single-image records returned by indexing: 'train'
    draws a random augmentation per sample, anything else gives the deterministic crop. The pool
    never depends on it. Records:
        input_data.image                     float32 tensor (3, H, W), ImageNet-normalized
        targets_data.contact_data.contact_h  float32 (778,)
        meta_info.sample_id (str), meta_info.mano_valid (float32 (1,))
    """
    frame_digits = None
    frame_ext = None

    def __init__(self, data_split='train', input_size=None):
        super().__init__()
        self.data_split = data_split
        self.input_size = input_hw(input_size)

    def __len__(self):
        return len(self.split)

    def sample_id(self, i):
        raise NotImplementedError

    def target(self, i):
        raise NotImplementedError

    def frame_path(self, seq_dir, frame):
        return os.path.join(seq_dir, f'{int(frame):0{self.frame_digits}d}{self.frame_ext}')

    def __getitem__(self, i):
        t = self.target(i)
        img = read_rgb(t.image, t.mirror)
        aug = sample_aug(self.input_size) if self.data_split == 'train' else None
        M = box_to_crop(t.box, self.input_size, aug, img.shape[:2])
        image = to_input(crop_image(img, M, self.input_size, aug))
        return {'input_data': {'image': image},
                'targets_data': {'contact_data': {'contact_h': t.contact}},
                'meta_info': {'sample_id': self.sample_id(i),
                              'mano_valid': np.ones(1, np.float32)}}
