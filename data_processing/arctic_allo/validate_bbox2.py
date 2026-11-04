"""Reproduce HACO's cam-0 bbox_hand_r exactly, now that we know ARCTIC's two mappings.

ARCTIC (scripts_data/crop_images.py::process_fname):
  view 0 (ego)  : cropped = FULL image resized 2800x2000 -> 840x600   (flat x0.3)
  view 1..8     : cropped = crop around bbox_loose (scale*1.5) -> 1000x1000
                  == common/data_utils.py::transform_kp2d

So the cam-0 mapping is just a scale. Find (a) which of the 10 view slots in 2d['joints.right'] is
the ego camera, and (b) what HACO put around the keypoints to get bbox_hand_r.
"""
import numpy as np
import glob, os, re

HACO_ANNOT = 'data/ARCTIC/preprocessed_data/train/annot_data'
SEQ = 's01/box_grab_01'

print("loading p2_train.npy ...", flush=True)
p2 = np.load('data/arctic_allo_work/p2_train.npy', allow_pickle=True).item()
e = p2['data_dict'][SEQ]
j2d = e['2d']['joints.right']            # (F, 10, 21, 2) ORIGINAL image coords
v2d = e['2d']['joints.smplx']            # (F, 10, 127, 2) -- more points, in case HACO used these

haco = {}
for f in glob.glob(os.path.join(HACO_ANNOT, 's01-box_grab_01-0-*.npz')):
    n = int(f.rsplit('-', 1)[-1].split('.')[0])
    haco[n] = np.load(f, allow_pickle=True)['bbox_hand_r'].astype(np.float64)
sane = sorted(n for n, b in haco.items() if 0 < b[2] < 800 and 0 < b[3] < 800)
OFF = 10                                  # frame 10 -> row 0 (verified: frames 10..721)
EGO_SCALE = 840.0 / 2800.0                # = 600/2000 = 0.3

def tight(p):
    return np.array([p[:, 0].min(), p[:, 1].min(),
                     p[:, 0].max() - p[:, 0].min(), p[:, 1].max() - p[:, 1].min()])

print(f"\n=== which view slot is the EGO camera? (map orig x{EGO_SCALE} -> 840x600) ===")
probe = sane[len(sane)//4: len(sane)//4 + 6]
scores = np.zeros(10)
for n in probe:
    r = n - OFF
    t = haco[n]
    ct = np.array([t[0] + t[2]/2, t[1] + t[3]/2])
    for v in range(10):
        b = tight(j2d[r, v] * EGO_SCALE)
        cb = np.array([b[0] + b[2]/2, b[1] + b[3]/2])
        scores[v] += np.linalg.norm(ct - cb)
best_v = int(np.argmin(scores))
for v in range(10):
    print(f"   view {v}: mean center-dist to HACO = {scores[v]/len(probe):8.1f} px"
          + ("   <== EGO" if v == best_v else ""))

print(f"\n=== with ego view = {best_v}: what expansion did HACO apply? ===")
rw, rh = [], []
for n in sane[::40][:40]:
    r = n - OFF
    t = haco[n]
    bj = tight(j2d[r, best_v] * EGO_SCALE)          # 21 joints
    bv = tight(v2d[r, best_v] * EGO_SCALE)          # 127 smplx pts
    rw.append((t[2]/max(bj[2],1e-6), t[3]/max(bj[3],1e-6),
               t[2]/max(bv[2],1e-6), t[3]/max(bv[3],1e-6)))
rw = np.array(rw)
print(f"   HACO/joints21  w={np.median(rw[:,0]):.3f}  h={np.median(rw[:,1]):.3f}")
print(f"   HACO/smplx127  w={np.median(rw[:,2]):.3f}  h={np.median(rw[:,3]):.3f}")

# center agreement (does HACO's box share the keypoint centroid?)
dc = []
for n in sane[::40][:40]:
    r = n - OFF
    t = haco[n]
    b = tight(j2d[r, best_v] * EGO_SCALE)
    dc.append(np.hypot((t[0]+t[2]/2)-(b[0]+b[2]/2), (t[1]+t[3]/2)-(b[1]+b[3]/2)))
print(f"   center distance HACO vs joint-box: median={np.median(dc):.1f} px  max={np.max(dc):.1f} px")
print(f"   (image is 840x600, so a few px == same box)")
