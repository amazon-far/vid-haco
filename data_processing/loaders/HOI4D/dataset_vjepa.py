"""
V-JEPA 2 (video) reader for HOI4D.

For each labeled HOI4D target frame `t` (sample_id = "<seq>-<frame>"), this builds a
T-frame RGB clip ENDING at `t` (T = clip_len, default 16; multidata_vjepa uses 8), crops every
frame with the SAME hand-crop affine so the clip is spatially aligned with the 778-vertex contact
GT defined for frame `t`, and returns:

    input_data.image      : [T, 3, H, W]  (V-JEPA-normalized RGB clip, oldest->newest;
                                           H, W = input_size, default 256 x 256)
    targets_data.contact_data.contact_h : [778] float32  (binary per-vertex GT for frame t)
    targets_data.hand_contact_gt         : [2]   float32  ([left, right] per-hand binary GT)

On-disk layout (labels from HACO's preprocessed release, frames from the official HOI4D release):
    data/HOI4D/preprocessed_data/train/annot_data/<sample_id>.npz    'bbox_ho' (x, y, w, h)
    data/HOI4D/preprocessed_data/train/contact_data/<sample_id>.npy  (778,)
    data/HOI4D/data/HOI4D_color[_half]/HOI4D_release/<seq with '/'>/align_rgb/<frame:05d>.jpg

------------------------------------------------------------------------------------
CLIP CONSTRUCTION (documented):
  We sample the T-1 frames PRECEDING `t` with a fixed stride `CLIP_STRIDE` (default 2),
  giving frame indices [t - (T-1)*stride, ..., t - stride, t]. At a sequence start we clamp
  negative indices to 0 (i.e. repeat the first available frame), so a clip is always
  exactly T frames and the LAST frame is always the labeled target `t`. HOI4D color runs
  ~15 fps, so stride 2 over 16 frames spans ~30 frames ~= 2 s of motion context.

SPATIAL ALIGNMENT (documented):
  The hand box + augmentation parameters are drawn ONCE for the target frame's annot
  (`bbox_ho`, see loaders/common.py). The resulting affine (and the flip / colour gain) are then
  applied IDENTICALLY to all T frames. This keeps the clip internally consistent and aligned to
  exactly the same crop in which the 778-vertex contact GT for frame `t` is defined, so the
  model's last-frame prediction matches the GT. Clip frames get no further degradations.

2-BIT PER-HAND GT (documented):
  HOI4D annot provides only {sample_id, bbox_ho} -- there is NO handedness field, and HOI4D
  is a (predominantly) single-hand dataset. We therefore set the per-hand binary GT as
  [left_bit, right_bit] = [0, float(contact_h.sum() > 0)]: the RIGHT hand is treated as the
  active hand (DEFAULT-TO-RIGHT rule, documented), its bit = 1 iff any of the 778 vertices is
  in contact, and the LEFT bit is always 0. (The 778-vertex contact head is the metric that is
  evaluated; the 2-bit head is an auxiliary "is the hand touching anything" signal.)

DETERMINISTIC 90/10 SPLIT BY SEQUENCE (documented):
  We take every sample_id that has BOTH an annot_data .npz and a contact_data .npy on disk
  (intersection of directory listings), group by SEQUENCE, sort the unique sequence names by
  md5(seq) hex (deterministic, independent of filesystem order), and hold out the last 10% of
  sequences as VAL. No val frame/clip shares a sequence with any train frame. The chosen
  train/val sample_id lists are cached to data/HOI4D/splits_vjepa/{train,val}.json so the split
  is reproducible and earlier checkpoints can be evaluated on the SAME val set.

No contact-balanced resampling is applied (it would drop all zero-contact frames and bias the
binary GT to always-positive).
"""
import os
import re
import json
import hashlib
import numpy as np

import torch

from ..common import (DATA_ROOT, RGB_MEAN, RGB_STD, box_to_crop, crop_image, input_hw, read_rgb,
                      sample_aug, to_input)


# ---- clip + split constants (these are DEFAULTS; the dataset reads instance attrs that the
#      trainer can override via the HOI4D_VJEPA(...) constructor or the env vars below) ----
CLIP_LEN = int(os.environ.get('VJEPA_FRAMES', 16))   # frames per clip (16/2 = 8 temporal tokens)
CLIP_STRIDE = int(os.environ.get('VJEPA_STRIDE', 2)) # stride between sampled frames
VAL_FRACTION = 0.10    # fraction of SEQUENCES held out for validation
SPLIT_DIR = os.path.join(DATA_ROOT, 'HOI4D', 'splits_vjepa')

# ---- resized-source config (the decode-speed optimization) ----
# Pre-resized frames live in a sibling tree: HOI4D_color -> HOI4D_color_half (see
# hoi4d/prep_resize_hoi4d.py). Reading those 960x540 JPEGs makes the per-clip JPEG decode ~4x
# cheaper so the GPUs stop starving. The bbox/affine are scaled by the SAME factor, so the final
# hand crop is near-identical to the full-res crop (measured mean-abs ~1.1/255,
# PSNR >40 dB, SSIM ~0.995). Default ON; toggle off with VJEPA_USE_RESIZED=0.
USE_RESIZED_DEFAULT = os.environ.get('VJEPA_USE_RESIZED', '1') not in ('0', 'false', 'False')
RESIZE_FACTOR_DEFAULT = float(os.environ.get('VJEPA_RESIZE_FACTOR', 0.5))
FULLRES_COLOR_TAG = 'HOI4D_color'        # the original full-res tree directory name
RESIZED_COLOR_TAG = 'HOI4D_color_half'   # the pre-resized sibling tree directory name

# V-JEPA 2 video preprocessing normalization (ImageNet mean/std on [0,1] RGB), matches
# transformers VJEPA2VideoProcessor defaults.
VJEPA_MEAN = RGB_MEAN
VJEPA_STD = RGB_STD


def _seq_of(sample_id):
    seq, _frame = re.match(r'(.+?)-(\d+)$', sample_id).groups()
    return seq


def build_or_load_split(verbose=True):
    """Build (or load cached) the deterministic 90/10 by-sequence split.

    Returns dict(train=[ids], val=[ids], n_seq_train, n_seq_val, n_seq_total).
    Cached to SPLIT_DIR/{train,val}.json. Computed identically on every process.
    """
    train_path = os.path.join(SPLIT_DIR, 'train.json')
    val_path = os.path.join(SPLIT_DIR, 'val.json')
    meta_path = os.path.join(SPLIT_DIR, 'split_meta.json')

    if os.path.isfile(train_path) and os.path.isfile(val_path):
        with open(train_path) as f:
            train_ids = json.load(f)
        with open(val_path) as f:
            val_ids = json.load(f)
        meta = {}
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        return dict(train=train_ids, val=val_ids,
                    n_seq_train=meta.get('n_seq_train'),
                    n_seq_val=meta.get('n_seq_val'),
                    n_seq_total=meta.get('n_seq_total'))

    root = os.path.join(DATA_ROOT, 'HOI4D')
    annot_dir = os.path.join(root, 'preprocessed_data', 'train', 'annot_data')
    contact_dir = os.path.join(root, 'preprocessed_data', 'train', 'contact_data')
    annot_ids = {f[:-4] for f in os.listdir(annot_dir) if f.endswith('.npz')}
    contact_ids = {f[:-4] for f in os.listdir(contact_dir) if f.endswith('.npy')}
    valid_ids = sorted(annot_ids & contact_ids)  # deterministic base ordering

    # group sample ids by sequence
    seq_to_ids = {}
    for sid in valid_ids:
        seq_to_ids.setdefault(_seq_of(sid), []).append(sid)

    # deterministic sequence ordering by md5 hash of the sequence name
    seqs = sorted(seq_to_ids.keys(), key=lambda s: hashlib.md5(s.encode()).hexdigest())
    n_total = len(seqs)
    n_val = max(1, int(round(n_total * VAL_FRACTION)))
    val_seqs = set(seqs[-n_val:])     # hold out the last 10% (by hash order)
    train_seqs = set(seqs[:-n_val])

    train_ids = sorted(sid for s in train_seqs for sid in seq_to_ids[s])
    val_ids = sorted(sid for s in val_seqs for sid in seq_to_ids[s])

    os.makedirs(SPLIT_DIR, exist_ok=True)
    with open(train_path, 'w') as f:
        json.dump(train_ids, f)
    with open(val_path, 'w') as f:
        json.dump(val_ids, f)
    meta = dict(n_seq_total=n_total, n_seq_train=len(train_seqs), n_seq_val=len(val_seqs),
                n_samples_train=len(train_ids), n_samples_val=len(val_ids),
                val_fraction=VAL_FRACTION, clip_len=CLIP_LEN, clip_stride=CLIP_STRIDE)
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    if verbose:
        print(f"[split] built by-sequence 90/10: seq total={n_total} train={len(train_seqs)} "
              f"val={len(val_seqs)} | samples train={len(train_ids)} val={len(val_ids)}", flush=True)
    return dict(train=train_ids, val=val_ids,
                n_seq_train=len(train_seqs), n_seq_val=len(val_seqs), n_seq_total=n_total)


class HOI4D_VJEPA(torch.utils.data.Dataset):
    """HOI4D as T-frame RGB clips for a frozen-V-JEPA-2 + DPT contact model.

    data_split in {'train','val'} selects the deterministic by-sequence split above. The
    training split draws one random crop augmentation per clip (common.sample_aug), the val
    split uses the deterministic crop (no random aug) for a stable metric.
    input_size: crop (height, width) or int; None -> common.DEFAULT_INPUT_SIZE (256 x 256).
    """

    def __init__(self, data_split='train', clip_len=None, clip_stride=None,
                 use_resized=None, resize_factor=None, input_size=None):
        super().__init__()
        assert data_split in ('train', 'val'), data_split
        self.data_split = data_split
        self.root_path = os.path.join(DATA_ROOT, 'HOI4D')
        self.data_dir = os.path.join(DATA_ROOT, 'HOI4D', 'data')
        self.annot_data_path = os.path.join(self.root_path, 'preprocessed_data', 'train', 'annot_data')
        self.contact_data_path = os.path.join(self.root_path, 'preprocessed_data', 'train', 'contact_data')
        self.input_size = input_hw(input_size)

        # ---- per-instance clip + source config (fall back to module defaults / env) ----
        self.clip_len = int(clip_len) if clip_len is not None else CLIP_LEN
        self.clip_stride = int(clip_stride) if clip_stride is not None else CLIP_STRIDE
        self.use_resized = USE_RESIZED_DEFAULT if use_resized is None else bool(use_resized)
        self.resize_factor = RESIZE_FACTOR_DEFAULT if resize_factor is None else float(resize_factor)
        # which on-disk color tree to read from
        self._color_tag = RESIZED_COLOR_TAG if self.use_resized else FULLRES_COLOR_TAG
        # bbox/affine scale to apply to the target-frame annotation so the crop is the
        # same region as full-res. 1.0 when reading full-res; resize_factor when reading resized.
        self._bbox_scale = self.resize_factor if self.use_resized else 1.0
        # if the resized tree is missing for a sequence, transparently fall back to full-res for
        # that sequence (keeps training robust mid-prep); recorded so we can warn once.
        self._fallback_warned = False

        split = build_or_load_split(verbose=False)
        self.split = split['train'] if data_split == 'train' else split['val']

        # cache per-sequence frame counts (so we know the max frame index to clamp to)
        self._seq_nframes = {}

    def __len__(self):
        return len(self.split)

    # ---------- helpers ----------
    def _frame_dir(self, seq_slash, color_tag=None):
        tag = color_tag if color_tag is not None else self._color_tag
        return os.path.join(self.data_dir, tag, 'HOI4D_release', seq_slash, 'align_rgb')

    def _resolve_frame_dir(self, seq_slash):
        """Return (frame_dir, bbox_scale). Prefer the configured (resized) tree; if it is
        absent for this sequence, fall back to full-res with bbox_scale=1.0."""
        if not self.use_resized:
            return self._frame_dir(seq_slash, FULLRES_COLOR_TAG), 1.0
        d = self._frame_dir(seq_slash, RESIZED_COLOR_TAG)
        if os.path.isdir(d):
            return d, self._bbox_scale
        # resized tree not present for this seq -> full-res fallback (still correct, just slower)
        if not self._fallback_warned:
            print(f"[dataset_vjepa] WARN resized dir missing ({d}); falling back to full-res "
                  f"for missing sequences", flush=True)
            self._fallback_warned = True
        return self._frame_dir(seq_slash, FULLRES_COLOR_TAG), 1.0

    def _nframes(self, frame_dir):
        if frame_dir not in self._seq_nframes:
            try:
                n = len([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
            except OSError:
                n = 0
            self._seq_nframes[frame_dir] = n
        return self._seq_nframes[frame_dir]

    def _clip_frame_indices(self, t, n_frames):
        """clip_len indices ending at t, stride clip_stride, clamped to [0, n_frames-1]."""
        idxs = [t - i * self.clip_stride for i in range(self.clip_len)][::-1]  # oldest .. t
        hi = max(0, n_frames - 1)
        idxs = [min(max(j, 0), hi) for j in idxs]
        return idxs

    def __getitem__(self, i):
        sample_id = str(self.split[i])
        seq_dash, frame_str = re.match(r'(.+?)-(\d+)$', sample_id).groups()
        seq_slash = seq_dash.replace('-', '/')
        t = int(frame_str)

        # resolve which color tree to read (resized vs full-res) + the matching bbox scale
        frame_dir, bbox_scale = self._resolve_frame_dir(seq_slash)
        n_frames = self._nframes(frame_dir)
        frame_idxs = self._clip_frame_indices(t, n_frames)

        # ----- labels (defined for target frame t) -----
        with np.load(os.path.join(self.annot_data_path, f'{sample_id}.npz'), allow_pickle=True) as annot:
            box = np.asarray(annot['bbox_ho'], dtype=np.float32)
        contact_h = np.load(os.path.join(self.contact_data_path, f'{sample_id}.npy')).astype(np.float32)  # [778]

        # 2-bit per-hand GT: [left, right] = [0, contact present]  (default-to-right rule)
        hand_contact_gt = np.array([0.0, float(contact_h.sum() > 0)], dtype=np.float32)  # [2]

        # The bbox annotation (bbox_ho) is in FULL-RES pixel coords. When reading the resized
        # tree, scale the bbox by the resize factor so it lands on the same hand region in the
        # smaller image -> the crop's destination (which depends only on the output size) is
        # unchanged, so the final crop is the SAME region as full-res, just resampled from a
        # smaller image. (Verified: mean-abs ~1.1/255, PSNR >40 dB.)
        if bbox_scale != 1.0:
            box = box * bbox_scale   # [x, y, w, h] all scale linearly

        # ----- crop/aug drawn ONCE for the TARGET frame, applied to all frames -----
        target_img = read_rgb(os.path.join(frame_dir, f'{t:05d}.jpg'))  # RGB float32 [H,W,3]
        aug = sample_aug(self.input_size) if self.data_split == 'train' else None
        M = box_to_crop(box, self.input_size, aug, target_img.shape[:2])

        # build the clip with the SAME affine for each frame
        out_h, out_w = self.input_size
        clip = torch.empty((self.clip_len, 3, out_h, out_w), dtype=torch.float32)
        decoded = {t: target_img}
        for k, fidx in enumerate(frame_idxs):
            if fidx not in decoded:
                decoded[fidx] = read_rgb(os.path.join(frame_dir, f'{fidx:05d}.jpg'))
            clip[k] = to_input(crop_image(decoded[fidx], M, self.input_size, aug, degrade=False))

        return {'input_data': {'image': clip},
                'targets_data': {'contact_data': {'contact_h': contact_h},
                                 'hand_contact_gt': hand_contact_gt},
                'meta_info': {'sample_id': sample_id, 'mano_valid': np.ones(1, np.float32)}}
