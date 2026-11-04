"""Decisive visual check: is the 778-contact label bound to the hand actually in the crop?

The earlier grid was rendered from the TRAIN split, whose crops go through the random augmentation
(scale, rotation, translation, colour gain) -- that can push the hand out of the crop and makes the
check meaningless. The VAL split uses the deterministic crop (scale=1, rot=0, no flip, no shift),
i.e. the clean crop. We render from VAL, and ALSO draw the stored bbox on the full upright frame.

Run from the directory that contains data/:
    HACO_NO_SEG=1 python <repo>/data_processing/hot3d_contact/haco_adapter/verify_visual.py

Expect: high-contact -> hand visibly gripping an object; zero-contact -> hand free / reaching.
"""
import os, sys, json
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))  # repo root
os.environ.setdefault('HACO_NO_SEG', '1')

OUT = 'data/hot3d_contact/haco_adapter/_verify_out'
FRAMES = 'data/HOT3D/data'
ANNOT = 'data/HOT3D/preprocessed_data/train/annot_data'
os.makedirs(OUT, exist_ok=True)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def denorm(chw):
    x = np.asarray(chw, np.float32).transpose(1, 2, 0)
    return cv2.cvtColor(np.clip((x * STD + MEAN) * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def full_with_bbox(sid, size=360):
    """Re-draw the stored bbox on the extracted upright frame (flipped for left hands)."""
    seq, hand, fr = sid.rsplit('-', 2)
    img = cv2.imread(os.path.join(FRAMES, seq, f'{int(fr):06d}.jpg'))
    if img is None:
        return np.zeros((size, size, 3), np.uint8)
    if hand == 'left':
        img = cv2.flip(img, 1)                       # exactly what the loader does
    b = np.load(os.path.join(ANNOT, sid + '.npz'), allow_pickle=True)['bbox_hand_r']
    x, y, w, h = [float(v) for v in b]
    c = (60, 255, 60) if hand == 'right' else (255, 140, 60)
    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), c, 6)
    img = cv2.resize(img, (size, size))
    cv2.rectangle(img, (0, 0), (size, 20), (0, 0, 0), -1)
    cv2.putText(img, f'{hand} bbox on full frame', (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
    return img


from data_processing.multidata_vjepa import MultiDataVJEPA

va = MultiDataVJEPA('val', clip_len=8, clip_stride=2, datasets=['HOT3D'], verbose=False)
print(f"val n={len(va)}  (test-mode aug -> clean deterministic crop)")

rng = np.random.RandomState(7)
probe = rng.choice(len(va), size=min(900, len(va)), replace=False)
buckets = {'HIGH': [], 'ZERO': [], 'LEFT': []}
for i in probe:
    r = va[int(i)]
    c = np.asarray(r['targets_data']['contact_data']['contact_h'])
    sid = r['meta_info']['sample_id']
    n = int(c.sum())
    if n > 150 and '-right-' in sid and len(buckets['HIGH']) < 4:
        buckets['HIGH'].append((r, sid, n))
    elif n == 0 and len(buckets['ZERO']) < 4:
        buckets['ZERO'].append((r, sid, n))
    if n > 150 and '-left-' in sid and len(buckets['LEFT']) < 4:
        buckets['LEFT'].append((r, sid, n))
    if all(len(v) >= 4 for v in buckets.values()):
        break

for tag, items in buckets.items():
    if not items:
        continue
    rows = []
    for r, sid, n in items:
        im = r['input_data']['image']
        im = im.numpy() if torch.is_tensor(im) else np.asarray(im)
        crop = cv2.resize(denorm(im[-1]), (360, 360))
        cv2.rectangle(crop, (0, 0), (360, 22), (0, 0, 0), -1)
        cv2.putText(crop, f'{tag}  n_contact={n}', (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1)
        rows.append(np.vstack([full_with_bbox(sid), crop]))
    cv2.imwrite(os.path.join(OUT, f'V_{tag}.jpg'), np.hstack(rows))
    print(f"  wrote V_{tag}.jpg   {[s for _, s, _ in items]}")
print(f"-> {OUT}")
