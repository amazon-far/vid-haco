"""End-to-end verification that HOT3D is trainable through the multi-dataset loader (multidata_vjepa).

Run from the directory that contains data/:
    HACO_NO_SEG=1 python <repo>/data_processing/hot3d_contact/haco_adapter/verify_hot3d.py

Checks
  1. MultiDataVJEPA(datasets=['HOT3D']) builds, train/val split is disjoint and ~90/10
  2. every record has the exact V-JEPA contract: image [T,3,256,256] float32, contact_h (778,),
     hand_contact_gt (2,)
  3. normalization really is ImageNet/V-JEPA (denormalized pixels land back in [0,255])
  4. VISUAL: crops sorted by contact -- high-contact samples must show a hand gripping an object,
     zero-contact samples must show a free hand. This is what proves the label is bound to the
     right image (the thing a shape check cannot prove).
  5. VISUAL: a clip's 8 frames really are consecutive video (motion), not a static repeat
  6. VISUAL: left-hand samples come out looking like RIGHT hands (the flip_extra path)
"""
import os, sys, json
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))  # repo root
os.environ.setdefault('HACO_NO_SEG', '1')

OUT = 'data/hot3d_contact/haco_adapter/_verify_out'
os.makedirs(OUT, exist_ok=True)

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def denorm(chw):
    """[3,256,256] normalized -> HxWx3 uint8 BGR for cv2."""
    x = np.asarray(chw, np.float32).transpose(1, 2, 0)
    x = (x * STD + MEAN) * 255.0
    return cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def label(img, txt, color=(0, 255, 255)):
    cv2.rectangle(img, (0, 0), (256, 22), (0, 0, 0), -1)
    cv2.putText(img, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return img


from data_processing.multidata_vjepa import MultiDataVJEPA

print("=== 1. build ===")
tr = MultiDataVJEPA('train', clip_len=8, clip_stride=2, datasets=['HOT3D'], verbose=True)
va = MultiDataVJEPA('val',   clip_len=8, clip_stride=2, datasets=['HOT3D'], verbose=True)
n_tr, n_va = len(tr), len(va)
print(f"  train={n_tr}  val={n_va}  val_frac={n_va/max(n_tr+n_va,1):.3f}  (expect ~0.10)")
assert n_tr > 0 and n_va > 0

print("\n=== 2/3. record contract + normalization ===")
rec = tr[0]
img = rec['input_data']['image']
con = rec['targets_data']['contact_data']['contact_h']
hcg = rec['targets_data']['hand_contact_gt']
print(f"  image {tuple(img.shape)} {img.dtype}   contact_h {tuple(np.shape(con))}   hand_gt {np.asarray(hcg)}")
assert tuple(img.shape) == (8, 3, 256, 256), img.shape
assert np.shape(con) == (778,), np.shape(con)
assert np.shape(hcg) == (2,)
d0 = denorm(img[-1].numpy() if torch.is_tensor(img) else img[-1])
print(f"  denormalized last frame: min={d0.min()} max={d0.max()} mean={d0.mean():.1f}  (expect ~[0,255])")
assert d0.max() > 100, "denormalized image looks black -- normalization mismatch"

# sample-id disjointness between train and val (the md5 hold-out)
ids_tr = {tr[i]['meta_info']['sample_id'] for i in np.random.RandomState(0).randint(0, n_tr, 300)}
ids_va = {va[i]['meta_info']['sample_id'] for i in np.random.RandomState(1).randint(0, n_va, 300)}
print(f"  train/val sample_id overlap: {len(ids_tr & ids_va)}  (expect 0)")
assert not (ids_tr & ids_va)

print("\n=== 4. scan for high/zero-contact + left-hand samples ===")
rng = np.random.RandomState(0)
probe = rng.choice(n_tr, size=min(600, n_tr), replace=False)
hi, zero, lefts = [], [], []
for i in probe:
    r = tr[int(i)]
    c = np.asarray(r['targets_data']['contact_data']['contact_h'])
    sid = r['meta_info']['sample_id']
    n = int(c.sum())
    if n > 120 and len(hi) < 4:
        hi.append((r, sid, n))
    elif n == 0 and len(zero) < 4:
        zero.append((r, sid, n))
    if '-left-' in sid and n > 60 and len(lefts) < 4:
        lefts.append((r, sid, n))
    if len(hi) >= 4 and len(zero) >= 4 and len(lefts) >= 4:
        break
print(f"  found high-contact={len(hi)} zero-contact={len(zero)} left-hand={len(lefts)}")

def strip(items, fname, tag):
    if not items:
        return
    tiles = []
    for r, sid, n in items:
        im = r['input_data']['image']
        t = denorm(im[-1].numpy() if torch.is_tensor(im) else im[-1])
        tiles.append(label(t, f"{tag} n_contact={n}"))
    cv2.imwrite(os.path.join(OUT, fname), np.hstack(tiles))
    print(f"  wrote {fname}  ({[s for _, s, _ in items][:2]} ...)")

strip(hi,    'A_high_contact.jpg', 'HIGH')
strip(zero,  'B_zero_contact.jpg', 'ZERO')
strip(lefts, 'D_left_hands.jpg',   'LEFT->flip')

print("\n=== 5. clip is real video (8 consecutive frames) ===")
r = hi[0][0] if hi else tr[0]
im = r['input_data']['image']
im = im.numpy() if torch.is_tensor(im) else np.asarray(im)
frames = [label(denorm(im[k]), f"t{k-7}") for k in range(8)]
cv2.imwrite(os.path.join(OUT, 'C_clip_strip.jpg'), np.hstack(frames))
diffs = [float(np.abs(im[k].astype(np.float32) - im[k+1].astype(np.float32)).mean()) for k in range(7)]
print(f"  mean |frame_t - frame_t+1| across the clip: {np.mean(diffs):.4f}")
print(f"  -> {'REAL MOTION (not a static repeat)' if np.mean(diffs) > 1e-4 else 'STATIC (bad!)'}")
print(f"  wrote C_clip_strip.jpg")

print(f"\n=== done -> {OUT} ===")
