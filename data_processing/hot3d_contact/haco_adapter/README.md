# HOT3D → per-sample training layout

Makes `data/hot3d_contact` (labels from `../pipeline`) + `data/hot3d_dataset` (official HOT3D download)
**directly trainable** by the V-JEPA contact trainer, through the HOT3D reader
[`loaders/HOT3D/dataset.py`](../../loaders/HOT3D/dataset.py).

**Status: done and verified — trains end-to-end** (loss 1.14 → 0.36 over 5k iterations).

```
HOT3D = 715,772 samples   (643,980 train / 71,792 val, deterministic md5 hold-out)
        136 Aria sequences, 476,709 upright RGB frames, 159 GB
        63.8% of samples have contact > 0
```

## Why it wasn't trainable before

| Needed by the trainer | HOT3D shipped |
|---|---|
| RGB frames as numbered per-seq dirs | only `recording.vrs` (raw sensor container) |
| hand bbox per frame | `box2d_hands.csv` — but in RAW VRS coords, different time domain |
| contact as per-sample `<sid>.npy` | per-sequence `.npz` of shape `(T,778)` |
| a registered reader | none |

## What was resolved (empirically, see `probe_alignment.py` / `probe_upright.py`)

1. **Only Aria has RGB.** Quest3 in HOT3D has *no* RGB camera — two 640×512 **monochrome** SLAM
   cams only, with tiny hand boxes (~90×45 px). So the RGB source is Aria `camera-rgb` (stream
   `214-1`, 1408×1408, 30 fps). Quest is deliberately **not** wired up; mixing grayscale 640×512
   into an ImageNet-normalized RGB model would be a domain shift, not free data.

2. **Two time domains.** VRS `capture_timestamp_ns` is **DEVICE** time (9.17e12); every HOT3D
   annotation (box2d, masks, `dynamic_objects.csv`, and hence our contact npz) is **TIMECODE** time
   (5.77e13). They share a clock rate but not an epoch — zero raw overlap. We convert with
   `provider.convert_from_device_time_to_timecode_ns()` and nearest-match (residual ≤ **1 ns**;
   frames are 33 ms apart, so this is unambiguous). Exact integer equality does *not* hold.

3. **box2d is in RAW VRS pixel coords**, and raw Aria RGB is rotated 90° CW from upright. Every
   other training dataset is upright, so we extract upright (`rot90 CCW`) and rotate the boxes:
   `(x0,y0,x1,y1) -> (y0, W-1-x1, y1, W-1-x0)`. Verified by drawing the rotated box on the rotated
   frame — it lands on the hand.

4. **Left hands** are emitted as PROX-style mirrored samples: the reader horizontally flips the
   whole frame (turning a left hand into a right-hand-looking crop) and the bbox is stored
   *pre-flipped in x*. The 778 contact vector is used **as-is** — MANO left/right share vertex
   ordering. Verified visually.

## Pipeline

Run from the directory that contains `data/` (`<repo>` = this repository):

```bash
# 0. one-time: isolated venv (do NOT install projectaria_tools into the training env — it would
#    fight torch/numpy).
python -m venv data/hot3d_contact/.venv_vrs
data/hot3d_contact/.venv_vrs/bin/pip install projectaria_tools opencv-python-headless pillow numpy
V=data/hot3d_contact/.venv_vrs/bin/python

# 1. VRS -> upright JPEG frame trees  (~35 min on 48 workers, 153 GB)
$V <repo>/data_processing/hot3d_contact/haco_adapter/extract_frames.py --workers 48 --quality 90
#    -> data/HOT3D/data/<seq>/%06d.jpg + frames_index.json

# 2. contact npz + box2d + quality masks -> preprocessed_data + splits  (~2 min)
$V <repo>/data_processing/hot3d_contact/haco_adapter/build_haco_hot3d.py --workers 48
#    -> HOT3D/preprocessed_data/train/{annot_data,contact_data}/<sid>.{npz,npy}
#    -> HOT3D/splits/train.json, HOT3D/dataset_stats.json
```

Both are **idempotent/resumable** (`.done` markers; re-running skips finished sequences).

Filters applied in step 2 (tunable): hand annotated in the contact npz, hand has a non-empty box on
the RGB stream, `visibility_ratio ≥ 0.5`, HOT3D `mask_good_exposure`, tight box ≥ 40 px.
Boxes are expanded to a **square** with 1.3× context (the crop stretches `bbox_w × bbox_h` onto the
square network input, so a square box avoids anisotropic stretch; the other datasets ship
roughly-square boxes with context).

## Training

HOT3D is registered in `data_processing/multidata_vjepa.py` (`READERS`, `SEQ_CLIP_DATASETS`) and gets
**real 8-frame neighbour clips**, not static repeats. It is part of the default `ALL_DATASETS` roster;
`export HACO_NO_HOT3D=1` drops it, or pass an explicit roster:

```python
MultiDataVJEPA('train', datasets=['HOT3D'])                     # HOT3D alone
MultiDataVJEPA('train', datasets=['HOI4D', 'ARCTIC', 'InterHand26M', 'HIC', 'MOW', 'H2O', 'PROX',
                                  'Decaf', 'Hi4D', 'epic_contact'])   # 10-dataset baseline, no HOT3D
```

## Verification

```bash
# from the directory that contains data/
HACO_NO_SEG=1 python <repo>/data_processing/hot3d_contact/haco_adapter/verify_hot3d.py    # contract + splits + motion
HACO_NO_SEG=1 python <repo>/data_processing/hot3d_contact/haco_adapter/verify_visual.py   # label<->image binding
```

`verify_visual.py` renders from the **val** split on purpose: train-split crops are randomly scaled,
rotated and shifted, which can move the hand out of view and makes a visual check meaningless. Val uses
the deterministic crop.

Results (`_verify_out/`): high-contact crops show hands gripping objects (mug, bottle, milk carton);
zero-contact crops show free/reaching hands; left-hand crops come out looking like right hands; the
8-frame clip shows real motion (mean inter-frame Δ = 0.20, not a static repeat).

## ⚠️ Known label characteristic (not an adapter bug)

Contact is computed **only against HOT3D's tracked object library** (~5.9 rigid objects per
sequence: mugs, keyboard, mouse, cellphone, dumbbell…). A hand gripping an **untracked** thing — a
pen, the whiteboard, the table, the person's own body — is labeled **zero contact**. We hit a clear
example in verification: `P0009_911a4ba5-right-000440` shows a hand holding a whiteboard marker with
`n_contact=0`.

This is inherited from the HOT3D annotations + the labeling rule (hand-vs-tracked-object distance),
not from this adapter. It means the zero-contact class carries some false negatives.
If that matters for your loss, consider restricting to frames where a tracked object is near the
hand — the per-sequence npz stores `{left,right}_min_dist` (778, float16, capped 0.2 m), so you can
threshold on "nearest tracked object within X cm" without recomputing anything.

## Files

| file | role |
|---|---|
| `probe_alignment.py` | settles streams / time domains / raw-vs-upright box coords |
| `probe_upright.py` | verifies the box-rotation formula + left-flip |
| `extract_frames.py` | VRS → upright JPEG frame trees + `frames_index.json` |
| `build_haco_hot3d.py` | contact npz + box2d + masks → per-sample `preprocessed_data` + splits |
| `verify_hot3d.py` | dataloader contract, split disjointness, real-motion check |
| `verify_visual.py` | label↔image binding (clean val crops + bbox on full frame) |

The reader is [`data_processing/loaders/HOT3D/dataset.py`](../../loaders/HOT3D/dataset.py).
