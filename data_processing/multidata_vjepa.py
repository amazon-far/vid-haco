"""
Multi-dataset V-JEPA-2 clip dataset for Vid-HACO contact learning.

Produces, for EVERY sample, a uniform V-JEPA clip record:

    input_data.image                     : [T, 3, H, W]  (ImageNet/V-JEPA-normalized RGB)
    targets_data.contact_data.contact_h  : [778]  float32    (binary per-vertex contact GT)
    targets_data.hand_contact_gt         : [2]    float32    ([left, right] per-hand binary GT)
    targets_data.seg_mask                : [H, W] float32    (hand-object mask; all zeros, see SEG_DATASETS)
    meta_info.sample_id / mano_valid / dataset / has_seg  (+ dataset_id, added by MultiDataVJEPA)

H x W = `input_size` (default loaders.common.DEFAULT_INPUT_SIZE = 256 x 256; the V-JEPA 2.1
ViT-L/384 model is trained with input_size=(384, 384)).

USAGE
  Put the repository root on PYTHONPATH and run from the directory that contains data/<NAME>/
  (layouts: loaders/haco_labels.py and the other readers' docstrings):
      from data_processing.multidata_vjepa import MultiDataVJEPA
      train = MultiDataVJEPA('train', clip_len=8, clip_stride=2, input_size=(384, 384))

READERS (data_processing/loaders/): each enumerates one dataset's labelled pool in a fixed order
and gives, per pool index, the sample id, the labelled frame, the hand box and the contact label.
  haco_labels.py           ARCTIC (egocentric), Decaf, Hi4D, InterHand26M, H2O, MOW, PROX, HIC
                           (labels from HACO's preprocessed release, images from the official ones)
  HOT3D/, ARCTIC_allo/, epic_contact/, HOI4D/dataset_vjepa.py   (labels built in this repository)

------------------------------------------------------------------------------------------------
CLIP CONSTRUCTION (documented)
  * HOI4D -> a REAL T-frame video clip ending at the labeled target frame, from
    loaders/HOI4D/dataset_vjepa.py::HOI4D_VJEPA (clip_len=T, stride, resized HOI4D_color_half tree,
    per-frame hand crop aligned to the target-frame contact GT). This is genuine temporal motion.
  * SEQ_CLIP_DATASETS (ARCTIC, ARCTIC_allo, H2O, Decaf, Hi4D, PROX, InterHand26M, HOT3D) are
    labelled per image but their frames sit in per-sequence directories, so they get the same
    kind of real clip: the clip_len frames ENDING at the labelled frame (stride clip_stride),
    every frame cropped with the labelled frame's affine.
  * MOW (standalone in-the-wild photos), HIC and EPIC-Contact (no neighbouring frames used) take
    the single hand crop and REPEAT it T times -> a STATIC clip [T,3,H,W]. RoPE V-JEPA accepts
    this; the "last temporal tubelet" the model reads is then just that frame. (A static clip is
    the correct no-motion limit: every frame identical -> the temporal conv sees a constant
    signal.)

AUGMENTATION (train split only; loaders/common.py::sample_aug)
  One draw per sample: scale, rotation, box-centre shift and per-channel colour gain are applied
  identically to every frame of a clip. Static clips (single images) also get the single-image
  degradations of that draw (low resolution, zoom, noise, motion blur); real clips do not.
  The val split always uses the deterministic crop.

NORMALIZATION
  All paths use ImageNet mean/std on [0,1] RGB == V-JEPA-2 video-processor default
  (VJEPA_MEAN/STD), so real clips and static repeats are on the SAME input scale.

2-BIT PER-HAND GT (documented)
  Every reader crops a RIGHT-HAND-centered box (bbox_ho / bbox_hand_r / bbox_hand; left hands
  are mirrored where the dataset has them) and its contact_h is the 778-vertex contact of that
  hand. So the per-hand GT is the DEFAULT-TO-RIGHT rule:
        hand_contact_gt = [left, right] = [0, float(contact_h.sum() > 0)].
  Note on ARCTIC: ARCTIC is physically BIMANUAL, but its labels are a right-hand crop +
  right-hand contact GT (bbox_hand_r), so within this single-crop contact task the
  active/labeled hand is the right hand -> same [0, contact>0] rule. (The 778 contact map is
  the evaluated metric; the 2-bit head is an auxiliary "is the (right) hand touching anything"
  signal.) HOI4D already encodes exactly this rule inside HOI4D_VJEPA.

HELD-OUT VALIDATION SPLIT (documented, deterministic ~10% per dataset)
  * HOI4D: BY-SEQUENCE (loaders/HOI4D/dataset_vjepa.py::build_or_load_split) -- val shares NO
    sequence with train. We use that exact cached split (clip_len=T).
  * The others: integer indices into the reader's pool, cached as
    SPLIT_DIR/<NAME>_{train,val}.json. The paper's files are data_processing/splits/; copy them
    into SPLIT_DIR to reuse them. A missing file is rebuilt BY-HASH(sample_id):
    md5(sample_id) mod 10 == 9 -> val (deterministic ~10%, independent of filesystem order).

No contact-balanced resampling is applied (it would drop all zero-contact frames, biasing the
binary GT positive).
"""
import hashlib
import json
import os
import re

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .loaders.common import (DATA_ROOT, RGB_MEAN, RGB_STD, box_to_crop, crop_image, crop_mask,
                             input_hw, read_rgb, sample_aug, to_input)
from .loaders.haco_labels import READERS as HACO_LABEL_READERS
from .loaders.ARCTIC_allo.dataset import ARCTIC_allo
from .loaders.HOI4D.dataset_vjepa import HOI4D_VJEPA
from .loaders.HOT3D.dataset import HOT3D
from .loaders.epic_contact.dataset import epic_contact

# name -> reader class (constructor: (transform, data_split, input_size=None))
READERS = dict(HACO_LABEL_READERS, HOT3D=HOT3D, ARCTIC_allo=ARCTIC_allo, epic_contact=epic_contact)

# Seg target: SEG_DATASETS gates any hand-object mask a reader puts in its Target. No reader supplies
# one -- HOI4D_VJEPA loads no masks and MOW's masks/both foreground only defines its crop box -- so
# seg_mask is all zeros and has_seg is False for every dataset, as in the released training code.
# NOTE: H2O's cam*/mask/ folder is NOT segmentation -- it is byte-identical to depth/ (a uint16
# depth map). H2O ships no hand-object seg, so it is EXCLUDED from seg supervision (has_seg=False
# always); it stays in the contact/binary roster. The other datasets have no masks either.
# HACO_NO_SEG=1 disables ALL mask loading (used by the contact-only run -> no seg anywhere).
SEG_DATASETS = set() if os.environ.get('HACO_NO_SEG') else {'HOI4D', 'MOW'}

# V-JEPA-2 video preprocessing normalization (== HOI4D_VJEPA / VJEPA2VideoProcessor defaults)
VJEPA_MEAN = RGB_MEAN
VJEPA_STD = RGB_STD

# Datasets labelled per image (see READERS). PROX is not in the default roster; the paper's
# 12-dataset mix is ALL_DATASETS + ['PROX'].
SINGLE_IMAGE_DATASETS = ['ARCTIC', 'HIC', 'MOW', 'H2O', 'InterHand26M', 'Decaf', 'Hi4D', 'epic_contact']

# HOT3D (Aria camera-rgb, egocentric video): real neighbour clips from the frames extracted out of
# recording.vrs, per-hand 778-vertex contact @1cm (hot3d_contact/). Included at STRIDE 1 (every
# labeled frame, no subsampling), the same policy as the other video datasets; it is simply a
# bigger dataset (~2-min recordings x 272 seq-hand pairs). The sampler is a plain shuffled
# DistributedSampler, so a dataset's share of gradient steps == its share of samples.
#
# ARCTIC_allo = ARCTIC's ALLOCENTRIC cameras (protocol p1). ARCTIC is protocol p2 = the
# EGOCENTRIC view (cam 0) only. Same frames, same view-independent 778-vertex contact label,
# extra viewpoints. Kept as a SEPARATE dataset so ARCTIC (and the 10-dataset baseline) is untouched.
#
# To reproduce the 10-dataset baseline (experiments_vjepa_g_dpt_8f_seq), pass an explicit roster:
#     --datasets HOI4D,ARCTIC,InterHand26M,HIC,MOW,H2O,PROX,Decaf,Hi4D,epic_contact
# or export HACO_NO_HOT3D=1 / HACO_NO_ARCTIC_ALLO=1 to drop them from the default roster.
OPTIONAL_DATASETS = []
if not os.environ.get('HACO_NO_HOT3D'):
    OPTIONAL_DATASETS.append('HOT3D')
if not os.environ.get('HACO_NO_ARCTIC_ALLO'):
    OPTIONAL_DATASETS.append('ARCTIC_allo')
ALL_DATASETS = ['HOI4D'] + SINGLE_IMAGE_DATASETS + OPTIONAL_DATASETS

# Datasets whose frames live on disk as per-sequence directories -> real clips ending at the
# labelled frame. The rest (MOW, HIC, epic_contact) use the static repeat.
SEQ_CLIP_DATASETS = {'ARCTIC', 'H2O', 'Decaf', 'Hi4D', 'PROX', 'InterHand26M', 'HOT3D',
                     'ARCTIC_allo'}

VAL_HASH_MOD = 10        # md5(sample_id) % 10
VAL_HASH_BUCKET = 9      # bucket held out for val  (-> ~10%)
SPLIT_DIR = os.path.join(DATA_ROOT, 'splits_multidata_vjepa')


# ------------------------------------------------------------------ helpers
def _hash_is_val(sample_id):
    """Deterministic ~10% hold-out by md5(sample_id). True -> validation."""
    h = hashlib.md5(str(sample_id).encode()).hexdigest()
    return (int(h[:8], 16) % VAL_HASH_MOD) == VAL_HASH_BUCKET


def _hand_contact_gt_from_contact(contact_h):
    """Default-to-right 2-bit GT: [left, right] = [0, contact present]."""
    present = float(np.asarray(contact_h).sum() > 0)
    return np.array([0.0, present], dtype=np.float32)


def _empty_seg(input_size=None):
    return np.zeros(input_hw(input_size), dtype=np.float32)


def _frame_order(name):
    """Listing order for irregularly named frames: first integer in the name, then the name."""
    m = re.search(r'\d+', name)
    return (int(m.group()) if m else 0, name)


# ------------------------------------------------------------------ one reader -> clip records
class _ClipDataset(Dataset):
    """One reader's labelled pool, restricted to this split's indices, as T-frame clip records.

    The reader is always built on its full pool; train and val are index subsets of it (cached
    in SPLIT_DIR). data_split='train' draws one random augmentation per sample, 'val' uses the
    deterministic crop. SEQ_CLIP_DATASETS get a real clip, the others a static repeat.
    """

    def __init__(self, name, data_split, clip_len=8, clip_stride=2, input_size=None):
        super().__init__()
        self.name = name
        self.data_split = data_split          # 'train' | 'val'
        self.clip_len = int(clip_len)
        self.clip_stride = int(clip_stride)
        self.input_size = input_hw(input_size)
        self.reader = READERS[name](None, 'train', input_size=self.input_size)
        self._n_base = len(self.reader)
        self._is_seq = name in SEQ_CLIP_DATASETS
        self._first_frame = {}   # seq_dir -> smallest frame number (numbered frames)
        self._listing = {}       # seq_dir -> ordered frame file names (irregular names)
        self.indices = self._build_split_indices()

    def _sample_id_for_index(self, i):
        return self.reader.sample_id(i)

    def _build_split_indices(self):
        cache = os.path.join(SPLIT_DIR, f"{self.name}_{self.data_split}.json")
        if os.path.isfile(cache):
            with open(cache) as f:
                return json.load(f)
        # compute sample_id per base index, bucket by hash
        train_idx, val_idx = [], []
        for i in range(self._n_base):
            sid = self._sample_id_for_index(i)
            (val_idx if _hash_is_val(sid) else train_idx).append(i)
        os.makedirs(SPLIT_DIR, exist_ok=True)
        # write both splits together (idempotent / deterministic)
        with open(os.path.join(SPLIT_DIR, f"{self.name}_train.json"), 'w') as f:
            json.dump(train_idx, f)
        with open(os.path.join(SPLIT_DIR, f"{self.name}_val.json"), 'w') as f:
            json.dump(val_idx, f)
        return train_idx if self.data_split == 'train' else val_idx

    def __len__(self):
        return len(self.indices)

    # ---- clip frames: clip_len paths, oldest .. labelled frame (last) ----
    def _neighbor_paths(self, t):
        seq_dir = os.path.dirname(t.image)
        back = [k * self.clip_stride for k in range(self.clip_len - 1, -1, -1)]
        if self.reader.frame_digits is not None:
            # numbered frames; a sequence may start at a nonzero frame -> clamp to its first one
            if seq_dir not in self._first_frame:
                try:
                    stems = [os.path.splitext(f)[0] for f in os.listdir(seq_dir)
                             if f.endswith(self.reader.frame_ext)]
                except OSError:
                    stems = []
                self._first_frame[seq_dir] = min(int(s) for s in stems) if stems else 0
            first = self._first_frame[seq_dir]
            return [self.reader.frame_path(seq_dir, min(max(t.frame - b, first), t.frame))
                    for b in back]
        # irregular / sparse names: neighbours in directory-listing order
        if seq_dir not in self._listing:
            try:
                names = [f for f in os.listdir(seq_dir) if f.lower().endswith(('.jpg', '.png'))]
            except OSError:
                names = []
            self._listing[seq_dir] = sorted(names, key=_frame_order)
        names = self._listing[seq_dir]
        name = os.path.basename(t.image)
        if name not in names:
            return [t.image] * self.clip_len
        pos = names.index(name)
        return [os.path.join(seq_dir, names[max(pos - b, 0)]) for b in back]

    def _video_clip(self, t, aug):
        """Real clip; crop and augmentation come from the labelled (last) frame."""
        paths = self._neighbor_paths(t)
        target = read_rgb(paths[-1], t.mirror)
        M = box_to_crop(t.box, self.input_size, aug, target.shape[:2])
        decoded = {paths[-1]: target}
        clip = torch.empty((self.clip_len, 3) + self.input_size, dtype=torch.float32)
        for k, path in enumerate(paths):
            if path not in decoded:
                try:
                    decoded[path] = read_rgb(path, t.mirror)
                except (IOError, OSError):
                    decoded[path] = target    # missing neighbour -> repeat target (rare gaps)
            clip[k] = to_input(crop_image(decoded[path], M, self.input_size, aug, degrade=False))
        return clip, target, M

    def _static_clip(self, t, aug):
        """Single augmented crop of the labelled frame, repeated clip_len times."""
        img = read_rgb(t.image, t.mirror)
        M = box_to_crop(t.box, self.input_size, aug, img.shape[:2])
        frame = to_input(crop_image(img, M, self.input_size, aug))
        return frame.unsqueeze(0).repeat(self.clip_len, 1, 1, 1), img, M

    def _seg(self, t, img, M, aug):
        """Seg target with the RGB's mirror + affine (+ zoom for static clips), if any."""
        if self.name not in SEG_DATASETS or t.mask is None:
            return _empty_seg(self.input_size), False
        mask = t.mask.astype(np.uint8)
        if t.mirror:
            mask = cv2.flip(mask, 1)
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        return crop_mask(mask, M, self.input_size, aug, degrade=not self._is_seq), True

    def _record(self, base_i):
        t = self.reader.target(base_i)
        aug = sample_aug(self.input_size) if self.data_split == 'train' else None
        build = self._video_clip if self._is_seq else self._static_clip
        clip, img, M = build(t, aug)
        seg_mask, has_seg = self._seg(t, img, M, aug)
        contact_h = np.asarray(t.contact, dtype=np.float32)
        return {'input_data': {'image': clip},
                'targets_data': {'contact_data': {'contact_h': contact_h},
                                 'hand_contact_gt': _hand_contact_gt_from_contact(contact_h),
                                 'seg_mask': seg_mask},
                'meta_info': {'sample_id': str(self.reader.sample_id(base_i)),
                              'mano_valid': np.ones(1, np.float32), 'dataset': self.name,
                              'has_seg': bool(has_seg)}}

    def __getitem__(self, i):
        # Resilient read: if a single underlying file is missing/corrupt (some datasets ship a
        # sparse raw-image tree), fall back to the next index rather than killing the worker.
        for attempt in range(8):
            base_i = self.indices[(i + attempt) % len(self.indices)]
            try:
                return self._record(base_i)
            except (IOError, OSError, ValueError, KeyError) as e:
                if attempt == 0:
                    print(f"[multidata] {self.name} read failed @base_i={base_i} ({e}); skipping",
                          flush=True)
        raise IOError(f"[multidata] {self.name}: 8 consecutive read failures near index {i}")


# ------------------------------------------------------------------ HOI4D real-clip wrapper
class _HOI4DClipDataset(Dataset):
    """Thin wrapper around HOI4D_VJEPA: tags each record with dataset='HOI4D' and adds the
    (empty) seg keys the other datasets carry, so every record has the same keys."""

    def __init__(self, data_split, clip_len=8, clip_stride=2,
                 use_resized=True, resize_factor=0.5, input_size=None):
        super().__init__()
        self.name = 'HOI4D'
        self.input_size = input_hw(input_size)
        self.ds = HOI4D_VJEPA(data_split, clip_len=clip_len, clip_stride=clip_stride,
                              use_resized=use_resized, resize_factor=resize_factor,
                              input_size=self.input_size)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        rec = self.ds[i]
        meta, targets = rec['meta_info'], rec['targets_data']
        meta['dataset'] = 'HOI4D'
        meta['sample_id'] = str(meta.get('sample_id', f'HOI4D-{i}'))
        targets['contact_data']['contact_h'] = np.asarray(
            targets['contact_data']['contact_h'], dtype=np.float32)
        targets.setdefault('seg_mask', _empty_seg(self.input_size))
        meta.setdefault('has_seg', False)
        return rec


# ------------------------------------------------------------------ ConcatDataset that carries
#                                                                     per-sample dataset tags
class MultiDataVJEPA(Dataset):
    """Concatenate the per-dataset V-JEPA clip datasets for one split ('train' or 'val').

    input_size: crop (height, width) or int for every dataset; None -> 256 x 256.

    Exposes:
      .per_dataset_counts : {name: n_samples}
      .dataset_of(global_index) -> name      (used by the trainer's per-dataset val metrics)
      .subsets / .names / .cum                for cumulative indexing
    """

    def __init__(self, data_split, clip_len=8, clip_stride=2,
                 use_resized=True, resize_factor=0.5, datasets=None, verbose=True,
                 input_size=None):
        super().__init__()
        assert data_split in ('train', 'val')
        self.data_split = data_split
        self.clip_len = int(clip_len)
        self.input_size = input_hw(input_size)
        names = list(datasets) if datasets is not None else list(ALL_DATASETS)

        self.subsets = []
        self.names = []
        for name in names:
            if name == 'HOI4D':
                ds = _HOI4DClipDataset(data_split, clip_len=clip_len, clip_stride=clip_stride,
                                       use_resized=use_resized, resize_factor=resize_factor,
                                       input_size=self.input_size)
            else:
                ds = _ClipDataset(name, data_split, clip_len=clip_len, clip_stride=clip_stride,
                                  input_size=self.input_size)
            if len(ds) == 0:
                if verbose:
                    print(f"[multidata] WARN {name} {data_split} has 0 samples -> skipped", flush=True)
                continue
            self.subsets.append(ds)
            self.names.append(name)

        # cumulative index map
        self.lengths = [len(s) for s in self.subsets]
        self.cum = np.cumsum([0] + self.lengths)  # cum[i]..cum[i+1] -> subset i
        self.per_dataset_counts = {n: l for n, l in zip(self.names, self.lengths)}

        if verbose:
            tot = int(self.cum[-1])
            print(f"[multidata] split={data_split} clip_len={clip_len} total={tot}", flush=True)
            for n, l in self.per_dataset_counts.items():
                print(f"[multidata]   {n:14s}: {l}", flush=True)

    def __len__(self):
        return int(self.cum[-1])

    def _locate(self, index):
        # subset index via binary search on cumulative sums
        s = int(np.searchsorted(self.cum, index, side='right') - 1)
        local = index - int(self.cum[s])
        return s, local

    def dataset_of(self, index):
        s, _ = self._locate(index)
        return self.names[s]

    def __getitem__(self, i):
        s, local = self._locate(i)
        rec = self.subsets[s][local]
        # carry an integer dataset id (so default collate keeps a tensor we can use for per-DS
        # val metrics without string collation headaches)
        rec['meta_info']['dataset_id'] = self.names.index(rec['meta_info'].get('dataset', self.names[s]))
        return rec


# convenience for trainers: stable dataset-name <-> id ordering
def dataset_name_list(datasets=None):
    return list(datasets) if datasets is not None else list(ALL_DATASETS)
