"""Probe 2: verify the UPRIGHT extraction (rot90 CCW) + the box2d coordinate rotation, and the
left-hand flip -> right-hand-looking crop.

Aria RGB is stored in VRS rotated 90deg CW from upright. Every other training dataset is upright, so we
extract upright and rotate the boxes to match. This verifies the rotation formula EMPIRICALLY.

np.rot90(a, k=1) (== cv2.ROTATE_90_COUNTERCLOCKWISE):  b[W-1-x, y] = a[y, x]
  -> a source pixel (x,y) lands at (x', y') = (y, W-1-x)
  -> bbox: x'min=ymin, x'max=ymax, y'min=W-1-xmax, y'max=W-1-xmin
"""
import os, sys, csv
import numpy as np
import cv2
from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId

SEQ = sys.argv[1] if len(sys.argv) > 1 else "P0001_10a27bf7"
SEQ_DIR = f"data/hot3d_dataset/aria/{SEQ}"
CONTACT = f"data/hot3d_contact/contact_data/aria/{SEQ}.npz"
OUT = "data/hot3d_contact/haco_adapter/_probe_out"
RGB_STREAM = "214-1"
os.makedirs(OUT, exist_ok=True)


def rot90ccw_box(x0, y0, x1, y1, W):
    """Map a bbox from RAW VRS coords to rot90-CCW (upright) coords. W = source width."""
    return y0, W - 1 - x1, y1, W - 1 - x0      # -> (nx0, ny0, nx1, ny1)


provider = data_provider.create_vrs_data_provider(os.path.join(SEQ_DIR, "recording.vrs"))
rgb = StreamId(RGB_STREAM)
n = provider.get_num_data(rgb)
vrs_tc = np.array([provider.convert_from_device_time_to_timecode_ns(
    int(provider.get_image_data_by_index(rgb, i)[1].capture_timestamp_ns)) for i in range(n)], np.int64)

rows = [r for r in csv.DictReader(open(os.path.join(SEQ_DIR, "box2d_hands.csv")))
        if r["stream_id"] == RGB_STREAM and r["x_min[pixel]"] != ""]
box = {}   # ts -> hand -> (x0,y0,x1,y1,vis)
for r in rows:
    box.setdefault(int(r["timestamp[ns]"]), {})[int(r["hand_index"])] = (
        float(r["x_min[pixel]"]), float(r["y_min[pixel]"]),
        float(r["x_max[pixel]"]), float(r["y_max[pixel]"]), float(r["visibility_ratio[%]"]))

d = np.load(CONTACT, allow_pickle=True)
con = {int(t): i for i, t in enumerate(d["timestamps_ns"])}
ts_all = np.array(sorted(box.keys()), np.int64)
j = np.searchsorted(ts_all, vrs_tc).clip(1, len(ts_all) - 1)
bi = np.where(np.abs(ts_all[j] - vrs_tc) < np.abs(ts_all[j - 1] - vrs_tc), j, j - 1)
ts2idx = {int(ts_all[bi[i]]): i for i in range(n)}

HAND = {0: ("left", "left_contact", "left_present"), 1: ("right", "right_contact", "right_present")}

# find a good frame per hand: high visibility + real contact
picks = {}
for hi, (nm, ck, pk) in HAND.items():
    best = None
    for ts, hb in box.items():
        if hi not in hb or ts not in con:
            continue
        vis = hb[hi][4]
        ci = con[ts]
        if not d[pk][ci]:
            continue
        nc = int(d[ck][ci].sum())
        if vis > 0.85 and nc > 40:
            score = vis * nc
            if best is None or score > best[0]:
                best = (score, ts, nc, vis)
    picks[hi] = best
    print(f"  hand={nm}: best frame ts={best[1] if best else None} "
          f"n_contact={best[2] if best else '-'} vis={best[3] if best else '-'}")

for hi, best in picks.items():
    if best is None:
        continue
    _, ts, nc, vis = best
    nm = HAND[hi][0]
    i = ts2idx[int(ts)]
    raw = cv2.cvtColor(provider.get_image_data_by_index(rgb, i)[0].to_numpy_array(), cv2.COLOR_RGB2BGR)
    H, W = raw.shape[:2]

    up = cv2.rotate(raw, cv2.ROTATE_90_COUNTERCLOCKWISE)     # UPRIGHT frame
    Hu, Wu = up.shape[:2]

    vis_img = up.copy()
    for h2, (x0, y0, x1, y1, v) in sorted(box[int(ts)].items()):
        nx0, ny0, nx1, ny1 = rot90ccw_box(x0, y0, x1, y1, W)
        c = (60, 255, 60) if h2 == 1 else (255, 120, 60)
        cv2.rectangle(vis_img, (int(nx0), int(ny0)), (int(nx1), int(ny1)), c, 4)
        cv2.putText(vis_img, f"{HAND[h2][0]} vis={v:.2f}", (int(nx0), max(0, int(ny0) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, c, 3)
    cv2.putText(vis_img, "UPRIGHT (rot90 CCW) + ROTATED box2d", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
    cv2.imwrite(f"{OUT}/UP_{nm}_full.jpg", cv2.resize(vis_img, (704, 704)))

    # roughly the hand crop the loader takes, in upright coords
    x0, y0, x1, y1, v = box[int(ts)][hi]
    nx0, ny0, nx1, ny1 = rot90ccw_box(x0, y0, x1, y1, W)
    src = up
    if hi == 0:                                  # LEFT -> horizontal flip => looks like a RIGHT hand
        src = cv2.flip(up, 1)
        nx0, nx1 = Wu - nx1, Wu - nx0
    pad = 0.3 * max(nx1 - nx0, ny1 - ny0)
    cx0, cy0 = int(max(0, nx0 - pad)), int(max(0, ny0 - pad))
    cx1, cy1 = int(min(Wu, nx1 + pad)), int(min(Hu, ny1 + pad))
    crop = cv2.resize(src[cy0:cy1, cx0:cx1], (256, 256))
    tag = "LEFT flipped -> should look RIGHT" if hi == 0 else "RIGHT (upright)"
    cv2.putText(crop, tag, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.imwrite(f"{OUT}/UP_{nm}_crop.jpg", crop)
    print(f"  wrote UP_{nm}_full.jpg / UP_{nm}_crop.jpg  (ts={ts}, contact={nc} verts, vis={vis:.2f})")
