# Data processing: contact labels for the 12 training datasets

Every dataset is reduced to the same per-sample format: an RGB frame, a hand bounding box, and a
**778-dim binary contact vector** over the MANO hand vertices (a vertex is in contact if it is within the
contact threshold of the other surface — object, scene, other hand, body or face). This is the label
representation of [HACO](https://github.com/dqj5182/HACO_RELEASE) (Jung & Lee, NeurIPS 2025). Vid-HACO
adds 8-frame clips (stride 2) ending at each labeled frame for the video datasets.

| Dataset | Contact | View | Train | Val | Where the labels come from |
|---|---|---|---:|---:|---|
| HOT3D | object | ego | 643,980 | 71,792 | **computed here** from HOT3D MANO + object poses — [`hot3d_contact/`](hot3d_contact) |
| ARCTIC, allocentric | object | exo | 459,867 | 51,186 | **built here** — ARCTIC's view-independent labels reused for cameras 1, 3, 6 — [`arctic_allo/`](arctic_allo) |
| ARCTIC, egocentric | object | ego | 168,343 | 18,707 | HACO preprocessed release |
| HOI4D | object | ego | 106,278 | 11,995 | HACO preprocessed release; clip reader + by-sequence split here — [`loaders/HOI4D`](loaders/HOI4D) |
| DECAF | face | exo | 58,829 | 6,493 | HACO preprocessed release |
| Hi4D | body | exo | 49,902 | 5,306 | HACO preprocessed release |
| InterHand2.6M | hand | exo | 33,439 | 3,731 | HACO preprocessed release |
| H2O | object | ego | 30,117 | 3,350 | HACO preprocessed release |
| EPIC-Contact | object | ego | 2,035 | 237 | converted from [EPIC-Contact](https://github.com/Sid2697/epic-contact) annotations — see below |
| MOW | object | web | 307 | 43 | HACO preprocessed release |
| PROX | scene | exo | 157 | 21 | HACO preprocessed release |
| HIC | hand, object | exo | 99 | 16 | HACO preprocessed release |
| **Total** | | | **1,553,353** | **172,877** | |

## Layout

```
multidata_vjepa.py        multi-dataset clip loader (MultiDataVJEPA): clips, augmentation, train/val split
loaders/                  one reader per dataset (all written for this repository)
  common.py                 image decoding, box -> crop affine, crop augmentation, mask warping
  haco_labels.py            ARCTIC (ego), DECAF, Hi4D, InterHand2.6M, H2O, MOW, PROX, HIC
  HOI4D/dataset_vjepa.py    real 8-frame HOI4D clips + by-sequence train/val split
  HOT3D/dataset.py          HOT3D (Aria RGB)
  ARCTIC_allo/dataset.py    ARCTIC allocentric cameras (reads the index.npz built by arctic_allo/)
  epic_contact/dataset.py   EPIC-Contact
splits/                   the exact train/val index lists used for the paper (counts in the table above)
hot3d_contact/            HOT3D contact-label pipeline + conversion into the per-sample layout
arctic_allo/              ARCTIC allocentric-view index builder and validators
hoi4d/prep_resize_hoi4d.py  half-resolution HOI4D frame tree used by the clip reader
```

## Using the loader

All paths are relative to the working directory, which must contain `data/<NAME>/` for every dataset
used. Put the repository root on `PYTHONPATH`:

```python
from data_processing.multidata_vjepa import MultiDataVJEPA, ALL_DATASETS
train = MultiDataVJEPA('train', clip_len=8, clip_stride=2, input_size=(384, 384))  # ViT-L/384
val = MultiDataVJEPA('val', clip_len=8, clip_stride=2, input_size=(384, 384))
paper_mix = MultiDataVJEPA('train', datasets=ALL_DATASETS + ['PROX'])               # all 12 datasets
```

Each record holds `input_data.image` `[T, 3, H, W]` (ImageNet-normalized), `targets_data.contact_data.contact_h`
`[778]`, `targets_data.hand_contact_gt` `[2]`, `targets_data.seg_mask` `[H, W]` and
`meta_info.{sample_id, mano_valid, dataset, dataset_id, has_seg}`. `input_size` defaults to 256 × 256
(`loaders/common.py: DEFAULT_INPUT_SIZE`; the V-JEPA 2 ViT-H/256 model); the ViT-L/384 model is trained at
384 × 384. The default roster `ALL_DATASETS` leaves out PROX; `HACO_NO_HOT3D=1` / `HACO_NO_ARCTIC_ALLO=1`
drop HOT3D / ARCTIC allocentric from it. No reader supplies a segmentation mask, so `seg_mask` is all
zeros and `has_seg` is False for every dataset, as in the released training code (`HACO_NO_SEG=1`
disables mask loading altogether).

**Splits.** Each reader enumerates its dataset's labelled pool in a fixed order (for the eight datasets in
`haco_labels.py`, the order of HACO's training split list; for the others, see the reader's docstring). The
files in [`splits/`](splits) are integer indices into exactly that order. `MultiDataVJEPA` reads
`data/splits_multidata_vjepa/<NAME>_{train,val}.json`; copy `splits/*.json` there to train on the paper's
splits. A missing file is rebuilt with `md5(sample_id) % 10 == 9 → val` and written there. HOI4D uses its own
by-sequence split (`data/HOI4D/splits_vjepa/`).

**Augmentation** (train split; `loaders/common.py`). One draw per sample, applied identically to every frame
of a clip: scale 1 ± 0.25, rotation (60 % of samples, up to ±60°), box-centre shift up to ±10 % of the image,
per-channel colour gain 0.8–1.2 — the same distribution as HACO's training crops. Single-image samples
(MOW, HIC, EPIC-Contact) also get HACO's single-image degradations (low resolution, zoom, noise, motion
blur); video clips do not. As in the paper's runs, the noise step clips the crop to [0, 1] instead of
[0, 255], so the ~30 % of single-image training crops that draw noise come out almost uniformly dark. No
random horizontal flip is used. Validation crops are deterministic: the box centre maps to the crop centre
and the box is stretched to fill the crop.

## 1. Datasets labelled by HACO's preprocessed release

ARCTIC (egocentric), DECAF, Hi4D, InterHand2.6M, H2O, MOW, PROX, HIC and HOI4D use the hand boxes,
contact labels and training split lists of HACO's preprocessed data release (Hugging Face
[`dqj5182/haco-data`](https://huggingface.co/datasets/dqj5182/haco-data)) together with each dataset's
official images; `docs/data_train.md` in the [HACO repository](https://github.com/dqj5182/HACO_RELEASE)
explains how to download both. The readers need only these files (`<sid>` = sample id):

| Dataset | Pool (index order) | Images | Labels (`preprocessed_data/train/`) |
|---|---|---|---|
| ARCTIC | `splits/train.json`: indices into `imgnames` of `data/splits/p2_train.npy` | `data/cropped_images/<subj>/<seq>/<cam>/<frame:05d>.jpg` | `annot_data/<sid>.npz` (`bbox_hand_r`), `contact_data/<sid>.npy` |
| Decaf | `splits/train.json`: `<seq>_<cam>_<frame>` | `data/train/images/<seq>/<cam>/<frame:05d>.jpg` | `db_bb/<sid>.npy` (x0, y0, x1, y1), `db_contact/<sid>.npy` |
| Hi4D | `splits/train.json` (frame paths) + `splits/train_pid.json` (person 0/1) | `data/<pair>/<action>/images/<cam>/<frame:06d>.jpg` | `annot_data/<sid>.npz` (`bbox_hand_r_<pid>`, `mano_r_contact_<pid>`) |
| H2O | `splits/train.json`: `<subject>/<seq>/<obj>/<cam>/rgb/<frame>.png` | `data/<subject>_ego/<seq>/<obj>/<cam>/rgb/<frame:06d>.png` | `annot_data/<sid>.npz` (`bbox_hand_r`), `contact_data/<sid>.npy` |
| InterHand26M | `splits/train.json`: annotation ids; tables from `annotations/train/InterHand2.6M_train_data.pkl` (or the official `…_data.json`) | `images/train/<file_name>` | `annot_data/<sid>.npz` (`bbox_hand`), `contact_data/<sid>.npy` |
| MOW | `splits/train.json`: image ids | `data/images/<id>.jpg`, `data/masks/both/<id>.jpg` | `contact_data/<id>.npy`; box = mask bounding box, grown |
| PROX | `splits/train.json`: `…/<seq>/Color/<frame>` | `data/quantitative/recordings/<seq>/Color/<frame>.jpg` (mirrored) | `annot_data/<sid>.npz` (`bbox_ho`), `contact_data/<sid>.npy` |
| HIC | `splits/train.json`: COCO-style `images` / `annotations` lists | `data/<file_name>` | `annot_data/<sid>.npz` (`bbox_hand`), `contact_data/<sid>.npy` |
| HOI4D | own by-sequence split of `annot_data` ∩ `contact_data` | `data/HOI4D_color_half/HOI4D_release/<seq>/align_rgb/<frame:05d>.jpg` | `annot_data/<sid>.npz` (`bbox_ho`), `contact_data/<sid>.npy` |

All paths are under `data/<NAME>/`; the docstrings in [`loaders/haco_labels.py`](loaders/haco_labels.py)
give the sample-id formats. Clips use the neighbouring frames in the same directory (numbered frames, or
the directory listing for InterHand2.6M and PROX).

## 2. HOT3D (computed here)

See [`hot3d_contact/README.md`](hot3d_contact/README.md). In short: for every annotated frame and present
hand, pose the MANO hand and the HOT3D object meshes in the world frame and mark hand vertices within
**1 cm** of any object surface (HACO's hand-object rule). Only HOT3D train participants have MANO
annotations. Then [`hot3d_contact/haco_adapter`](hot3d_contact/haco_adapter) extracts upright Aria RGB
frames from the VRS files, converts timestamps and hand boxes, and writes the per-sample layout that
[`loaders/HOT3D/dataset.py`](loaders/HOT3D/dataset.py) reads.

```bash
python data_processing/hot3d_contact/pipeline/run_contacts.py      # per-sequence (T,778) contact npz
python data_processing/hot3d_contact/pipeline/aggregate_contacts.py
python data_processing/hot3d_contact/haco_adapter/extract_frames.py
python data_processing/hot3d_contact/haco_adapter/build_haco_hot3d.py
```

## 3. ARCTIC allocentric (built here)

HACO's ARCTIC labels cover only the egocentric camera (protocol p2, cam 0). Contact is a property of the
3D hand–object geometry, so the same label holds in every view; only the bounding box is per-view.
[`arctic_allo/build_arctic_allo.py`](arctic_allo/build_arctic_allo.py) projects ARCTIC's 2D joints into
each allocentric camera, builds a 2× expanded box, drops off-image/degenerate boxes, and writes one
`index.npz`. The paper uses cameras 1, 3 and 6 (511,053 samples):

```bash
export ARCTIC_USERNAME=... ARCTIC_PASSWORD=...          # your ARCTIC account
python data_processing/arctic_allo/download_cams_1to8.py  # allocentric cropped images
bash   data_processing/arctic_allo/get_p1_split.sh        # protocol-p1 split (range request into splits.zip)
python data_processing/arctic_allo/build_arctic_allo.py --cams 1 3 6
python data_processing/arctic_allo/verify_arctic_allo.py
```

## 4. EPIC-Contact

Each sample is the manually annotated central frame of an EPIC-Grasps stable-grasp clip. The sparse
per-region annotation (`contact_vertices_sparse.json`: fingers, thumb, palm) is mapped to the 778-vertex
MANO topology, and the hand box is the posed `hand_mesh.obj` projected with the dataset's fixed pinhole
camera (f = 5000, principal point at the image centre). The reader documents the exact format.
**The conversion script itself is not included in this release.**

## 5. HOI4D clips

HOI4D's labels are per frame. [`loaders/HOI4D/dataset_vjepa.py`](loaders/HOI4D/dataset_vjepa.py)
returns real 8-frame clips ending at the labeled frame, from a half-resolution frame tree made by
[`hoi4d/prep_resize_hoi4d.py`](hoi4d/prep_resize_hoi4d.py), and splits train/val **by sequence** so
validation shares no sequence with training.

## Licenses

Neither the datasets nor HACO's preprocessed labels are redistributed here; each keeps its own license
and terms of use (most are research-only). Obtain them from their official sources.
