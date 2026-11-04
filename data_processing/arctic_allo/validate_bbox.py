"""Reproduce HACO's shipped ARCTIC cam-0 bbox_hand_r from ARCTIC's ORIGINAL p2_train.npy.

If we can regenerate HACO's numbers, the same derivation is trustworthy for the allocentric views
(p1, cams 1-8) -- which is the only thing standing between us and ARCTIC_allo.

Unknowns to pin down empirically:
  * which index of the 10-view axis in 2d['joints.right'] is the egocentric camera
  * whether 2d joints are in ORIGINAL-image coords or already in CROPPED-image coords
  * how frame number in the imgname maps to the array row
  * what padding/scale HACO applied around the 21 joints to get bbox_hand_r
"""
import numpy as np
import glob, os, re

HACO_ANNOT = 'data/ARCTIC/preprocessed_data/train/annot_data'

print("loading p2_train.npy (8.7 GB) ...", flush=True)
p2 = np.load('data/arctic_allo_work/p2_train.npy', allow_pickle=True).item()
dd, imn = p2['data_dict'], p2['imgnames']
print(f"  seqs={len(dd)} imgnames={len(imn)}", flush=True)

SEQ = 's01/box_grab_01'
e = dd[SEQ]
j2d = e['2d']['joints.right']          # (F, V, 21, 2)
bb = e['bbox']                          # (F, 9, 3)
F, V = j2d.shape[0], j2d.shape[1]
print(f"\n{SEQ}: frames={F} views_in_2d={V} bbox_views={bb.shape[1]}")

# ---- frame-number -> row mapping ------------------------------------------------
rows = [(i, x) for i, x in enumerate(imn) if f'/{SEQ}/' in x]
print(f"  imgnames for this seq: {len(rows)}   first={rows[0][1]}  last={rows[-1][1]}")
fnums = sorted(int(re.search(r'/(\d+)\.jpg$', x).group(1)) for _, x in rows)
print(f"  frame numbers: {fnums[0]} .. {fnums[-1]}  (n={len(fnums)})")
print(f"  => row = frame - {fnums[0]}   (assuming contiguous)")
OFF = fnums[0]

# ---- HACO's shipped bboxes for this seq, cam 0 -----------------------------------
haco = {}
for f in glob.glob(os.path.join(HACO_ANNOT, 's01-box_grab_01-0-*.npz')):
    n = int(f.rsplit('-', 1)[-1].split('.')[0])
    haco[n] = np.load(f, allow_pickle=True)['bbox_hand_r'].astype(np.float64)
print(f"  HACO shipped cam-0 bboxes for this seq: {len(haco)}")

# pick TYPICAL frames (HACO's ARCTIC has degenerate outliers; judge on sane ones)
sane = [n for n, b in haco.items() if 0 < b[2] < 800 and 0 < b[3] < 800]
sane.sort()
probe = sane[len(sane)//4:len(sane)//4+5]
print(f"  sane frames: {len(sane)}/{len(haco)}   probing {probe}")


def bbox_from_pts(p):
    x0, y0 = p[:, 0].min(), p[:, 1].min()
    x1, y1 = p[:, 0].max(), p[:, 1].max()
    return np.array([x0, y0, x1 - x0, y1 - y0])


print("\n=== hypothesis test: which view index == ego cam, and is 2d already cropped-coords? ===")
for n in probe[:2]:
    r = n - OFF
    tgt = haco[n]
    print(f"\n frame {n} (row {r})  HACO bbox_hand_r = [{tgt[0]:8.1f} {tgt[1]:8.1f} {tgt[2]:7.1f} {tgt[3]:7.1f}]")
    best = []
    for v in range(V):
        raw = bbox_from_pts(j2d[r, v])                     # tight box over 21 joints
        # score = how close the tight box is to HACO's (HACO may pad, so compare CENTER first)
        cx_t, cy_t = tgt[0] + tgt[2]/2, tgt[1] + tgt[3]/2
        cx_r, cy_r = raw[0] + raw[2]/2, raw[1] + raw[3]/2
        dc = np.hypot(cx_t - cx_r, cy_t - cy_r)
        best.append((dc, v, raw))
    best.sort()
    for dc, v, raw in best[:3]:
        ratio_w = tgt[2] / max(raw[2], 1e-6)
        ratio_h = tgt[3] / max(raw[3], 1e-6)
        print(f"   view {v}: tight=[{raw[0]:8.1f} {raw[1]:8.1f} {raw[2]:7.1f} {raw[3]:7.1f}] "
              f"center_dist={dc:7.1f}  HACO/tight w={ratio_w:.3f} h={ratio_h:.3f}")
