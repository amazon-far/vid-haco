"""End-to-end check that ARCTIC_allo trains through the multi-dataset loader (multidata_vjepa).

Run from the directory that contains data/:
    python <repo>/data_processing/arctic_allo/verify_arctic_allo.py

Rendered from the VAL split (deterministic crop); train-split single-image crops get random
zoom/blur/low-res and would hide the hand.

The claim being tested: the SAME 778-vertex contact label, taken from ARCTIC's cam-0 file, is
correct for all 8 allocentric views of that frame (contact is 3D hand-object geometry -> view
independent). So for one frame we render every camera side by side and print its contact count --
they must be IDENTICAL across views, and every crop must show the same grasp from a different angle.
"""
import os, sys
import numpy as np
import cv2, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # repo root
os.environ.setdefault('HACO_NO_SEG', '1')
OUT = 'data/arctic_allo_work/_viz'
os.makedirs(OUT, exist_ok=True)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def denorm(chw):
    x = np.asarray(chw, np.float32).transpose(1, 2, 0)
    return cv2.cvtColor(np.clip((x * STD + MEAN) * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


from data_processing.multidata_vjepa import MultiDataVJEPA
from data_processing.loaders.ARCTIC_allo.dataset import ARCTIC_allo

print("=== build ===")
tr = MultiDataVJEPA('train', clip_len=8, clip_stride=2, datasets=['ARCTIC_allo'], verbose=True)
va = MultiDataVJEPA('val',   clip_len=8, clip_stride=2, datasets=['ARCTIC_allo'], verbose=True)
print(f"  train={len(tr)}  val={len(va)}  val_frac={len(va)/(len(tr)+len(va)):.3f}")

rec = tr[0]
img = rec['input_data']['image']
con = np.asarray(rec['targets_data']['contact_data']['contact_h'])
print(f"\n=== contract ===")
print(f"  image {tuple(img.shape)} {img.dtype}  contact {con.shape}  hand_gt {rec['targets_data']['hand_contact_gt']}")
assert tuple(img.shape) == (8, 3, 256, 256) and con.shape == (778,)

# clip must be REAL motion (neighbour frames), not a static repeat
im = img.numpy() if torch.is_tensor(img) else np.asarray(img)
d = np.mean([np.abs(im[k] - im[k+1]).mean() for k in range(7)])
print(f"  mean |frame_t - frame_t+1| = {d:.4f} -> {'REAL MOTION' if d > 1e-4 else 'STATIC (bad)'}")

# ---- the view-independence claim: one frame, all 8 cams, identical contact ----
print("\n=== same frame across all 8 cameras: contact must be IDENTICAL ===")
# CRITICAL: render through a VAL-mode reader. A train-mode reader applies the random augmentation
# (zoom, translation, blur, low-res), which can blank the crop and makes this check meaningless.
base = ARCTIC_allo(None, 'val')           # deterministic crop
seq_ids, cams, frames = base.seq_ids, base.cams, base.frames
# find a (seq, frame) present in many cams with real contact
from collections import defaultdict
# index is ordered (seq, cam, frame), so the same frame across cams is FAR apart -> map ALL of it
bykey = defaultdict(list)
for i in range(len(frames)):
    bykey[(int(seq_ids[i]), int(frames[i]))].append(i)
full = [(k, v) for k, v in bykey.items() if len(v) >= 8]
print(f"  frames present in all 8 cams: {len(full)}")
cand = None
for k, idxs in full:
    c = np.load(os.path.join(base.contact_data_path,
                f"{base.seqs[k[0]].replace('/', '-')}-0-{k[1]:05d}.npy"))
    if c.sum() > 80:
        cand = (k, idxs, int(c.sum())); break

if cand:
    (sid_, fr), idxs, nc = cand
    seq = base.seqs[sid_]
    print(f"  {seq} frame {fr}: {len(idxs)} cams, contact={nc} verts (from cam-0 label)")
    tiles, counts = [], []
    for i in sorted(idxs, key=lambda j: int(cams[j])):
        r = base[i]                      # base loader -> single 256x256 crop
        cimg = r['input_data']['image']
        t = denorm(cimg.numpy() if torch.is_tensor(cimg) else cimg)
        n = int(np.asarray(r['targets_data']['contact_data']['contact_h']).sum())
        counts.append(n)
        cv2.rectangle(t, (0, 0), (256, 22), (0, 0, 0), -1)
        cv2.putText(t, f"cam{int(cams[i])}  contact={n}", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(t)
    cv2.imwrite(f'{OUT}/allo_sameframe_allcams.jpg', np.hstack(tiles))
    print(f"  contact per cam: {counts}")
    print(f"  all identical: {len(set(counts)) == 1}  <-- must be True (view-independent label)")
    print(f"  wrote {OUT}/allo_sameframe_allcams.jpg")
print("\n=== done ===")
