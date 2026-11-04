"""Is the frame->row mapping row = frame - 10, or row = frame?

A 10-frame (0.33 s) shift would put the bbox on where the hand WAS, which looks plausible in a static
grasp and wrong during motion -- exactly the failure seen at frame 51 for cams 4/7/8.

Settle it against ground truth: HACO's shipped cam-0 bbox_hand_r. Ego mapping is a flat x0.3
(2800x2000 -> 840x600), so we can compare directly. Whichever offset minimises the centre distance
to HACO's box is the true one. Test on FAST-MOVING frames, where the two hypotheses diverge most.
"""
import numpy as np, glob, os

HACO_ANNOT = 'data/ARCTIC/preprocessed_data/train/annot_data'
SEQ, SUBJ_SEQ = 's01/box_grab_01', 's01-box_grab_01'
EGO_SCALE = 0.3

p2 = np.load('data/arctic_allo_work/p2_train.npy', allow_pickle=True).item()
j2d = p2['data_dict'][SEQ]['2d']['joints.right']       # (F,10,21,2) original coords

haco = {}
for f in glob.glob(os.path.join(HACO_ANNOT, f'{SUBJ_SEQ}-0-*.npz')):
    n = int(f.rsplit('-', 1)[-1].split('.')[0])
    haco[n] = np.load(f, allow_pickle=True)['bbox_hand_r'].astype(np.float64)
sane = sorted(n for n, b in haco.items() if 0 < b[2] < 800 and 0 < b[3] < 800)


def centre(pts):
    return np.array([(pts[:, 0].min() + pts[:, 0].max()) / 2,
                     (pts[:, 1].min() + pts[:, 1].max()) / 2])


# rank frames by hand SPEED (row-to-row centre motion) -- the two hypotheses only differ where
# the hand actually moves
c_ego = np.array([centre(j2d[r, 0]) for r in range(j2d.shape[0])])
speed = np.r_[0, np.linalg.norm(np.diff(c_ego, axis=0), axis=1)]

for label, OFF in [('row = frame - 10', 10), ('row = frame', 0)]:
    for tag, pool in [('ALL sane frames', sane),
                      ('FASTEST 15% (hand moving)', None)]:
        if pool is None:
            sp = [(speed[n - 10] if 0 <= n - 10 < len(speed) else 0, n) for n in sane]
            sp.sort(reverse=True)
            pool = [n for _, n in sp[:max(1, len(sp) // 7)]]
        d = []
        for n in pool:
            r = n - OFF
            if not (0 <= r < j2d.shape[0]):
                continue
            t = haco[n]
            ct = np.array([t[0] + t[2] / 2, t[1] + t[3] / 2])
            cb = centre(j2d[r, 0] * EGO_SCALE)
            d.append(np.linalg.norm(ct - cb))
        d = np.array(d)
        print(f"  {label:18s} | {tag:26s} median={np.median(d):6.1f} px  mean={d.mean():6.1f} px")
    print()
