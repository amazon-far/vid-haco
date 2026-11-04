# HOT3D — Dense Hand Contact Maps (HACO-style)

Dense per-vertex **hand contact labels** for the [HOT3D](https://facebookresearch.github.io/hot3d/)
egocentric hand-object dataset, computed with the labeling method of **HACO**
(*Learning Dense Hand Contact Estimation from Imbalanced Data*, Jung & Lee, arXiv:2505.11152),
using a **1 cm** distance threshold.

Generated: 2026-07-10.

## What a contact map is

For every annotated frame and every present hand, a **binary vector of length 778**
(one bit per MANO hand vertex) marks whether that hand vertex is within **1 cm** of any
object surface — i.e. in contact. This is HACO's label representation and rule, with HACO's
hand-object threshold of 1 cm.

## Method (HACO's labeling rule)

Per frame, per hand:
1. **Hand mesh** (778 verts) from HOT3D's MANO annotations, using HOT3D's own convention
   (MANO via `pipeline/mano_layer.py`, `use_pca=True`, `num_pca_comps=15`; wrist SE3 → axis-angle global-orient + translation),
   posed in the sequence **world frame** (meters).
2. **Object meshes** from the HOT3D object library (`.glb`), posed to the world frame with the
   per-frame `dynamic_objects.csv` pose (`t_wo`, `q_wo`).
3. **Contact**: for each hand vertex, the Euclidean distance to the nearest
   object surface point; `contact = dist < 0.01 m`. Across multiple present objects we take the
   per-vertex minimum distance (≡ OR of per-object contact at the same threshold).

Implementation note: the exact point-to-surface distance (`trimesh.proximity.ProximityQuery.on_surface`)
is replaced by a `scipy.cKDTree` over a dense surface sampling (~1.2 mm spacing) of each object, queried after
transforming the hand into the object's canonical frame (distance is rigid-invariant, so the object's
KD-tree is built once and cached). Validated against exact `on_surface`:
**99.95 % per-vertex label agreement, mean distance error 0.34 mm** — negligible vs the 10 mm threshold,
and ~300× faster.

Only **train-set** sequences carry hand ground-truth; the 6 test participants (P0004/5/6/8/16/20) have
no MANO hand annotations and are skipped (no hand → no hand contact).

## Layout

```
contact_data/
  aria/<seq>.npz          per-sequence contact (Aria)
  quest/<seq>.npz         per-sequence contact (Quest)
  contact_data_all.npy    (N,778) uint8  all hand-instances stacked (aggregate format)
  contact_means.npy       (778,)  float64 per-vertex contact frequency
  manifest.csv            row -> seq, hand, timestamp_ns, n_contact_verts
  dataset_stats.json      summary statistics
viz/                      visualizations (see gallery_index.json)
pipeline/                 code (hot3d_contact_lib.py, run_contacts.py, aggregate_contacts.py, viz_*.py)
```

### Per-sequence `.npz` keys

| key | shape | dtype | meaning |
|---|---|---|---|
| `timestamps_ns` | (T,) | int64 | frame timestamps |
| `{left,right}_present` | (T,) | bool | hand annotated at this frame |
| `{left,right}_contact` | (T,778) | uint8 | **binary contact @1cm** (the label) |
| `{left,right}_min_dist` | (T,778) | float16 | hand-vertex → nearest object surface (m, capped 0.2); enables re-thresholding |
| `{left,right}_nearest_obj` | (T,778) | int8 | index into `object_uids` of the touched object (−1 if none) |
| `object_uids` / `object_names` | (O,) | str | dynamic objects appearing in the sequence |
| `meta` | — | json | seq, headset, threshold, MANO config, per-hand stats |

`left` = HOT3D hand index 0, `right` = 1.

## Loading

```python
import numpy as np
d = np.load("contact_data/aria/P0001_10a27bf7.npz", allow_pickle=True)
contact = d["right_contact"]        # (T,778) uint8, 1 = vertex in contact @1cm
ts      = d["timestamps_ns"]
# re-threshold at e.g. 5mm from stored distances:
contact_5mm = (d["right_min_dist"].astype(np.float32) < 0.005).astype(np.uint8)
```

The per-vertex order matches MANO (778 verts, `MANO_RIGHT/LEFT.pkl`);
[`haco_adapter/`](haco_adapter) turns these files into the per-sample layout read by
`data_processing/loaders/HOT3D/dataset.py`.
