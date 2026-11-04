"""Probe: does box2d_hands.csv align with the RAW VRS RGB frame, and do timestamps match?

Answers, empirically, the 4 unknowns that determine the whole adapter design:
  (a) which VRS stream is the Aria RGB camera, and its frame count / timestamp domain
  (b) do VRS capture_timestamp_ns EXACTLY match box2d_hands.csv / contact-npz timestamps_ns
  (c) are box2d pixel coords in RAW VRS orientation, or an upright-rotated one
  (d) does a horizontal flip turn a left hand into a right-hand-looking crop (for flip_extra)

Writes overlay PNGs to OUT for visual confirmation.
"""
import os, sys, csv, json
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

# ---------------------------------------------------------------- (a) streams
provider = data_provider.create_vrs_data_provider(os.path.join(SEQ_DIR, "recording.vrs"))
print("=== (a) VRS streams ===")
for sid in provider.get_all_streams():
    label = provider.get_label_from_stream_id(sid)
    n = provider.get_num_data(sid)
    print(f"  {str(sid):10s} label={label:22s} n_data={n}")

rgb = StreamId(RGB_STREAM)
n_rgb = provider.get_num_data(rgb)
img0, rec0 = provider.get_image_data_by_index(rgb, 0)
arr0 = img0.to_numpy_array()
print(f"\n  RGB stream {RGB_STREAM}: {n_rgb} frames, frame0 shape={arr0.shape} dtype={arr0.dtype}")

# all VRS timestamps for the RGB stream.
# NOTE: capture_timestamp_ns is DEVICE time; HOT3D's box2d/contact CSVs are in TIMECODE time
# (that is what timecode_devicetime_mapping.csv exists for). Convert, then nearest-match --
# the mapping rounds by ~1ns so exact integer equality does NOT hold.
vrs_dev = np.array([provider.get_image_data_by_index(rgb, i)[1].capture_timestamp_ns
                    for i in range(n_rgb)], dtype=np.int64)
vrs_ts = np.array([provider.convert_from_device_time_to_timecode_ns(int(t)) for t in vrs_dev],
                  dtype=np.int64)
print(f"  vrs device ts[:3]={vrs_dev[:3]}  dt_median={np.median(np.diff(vrs_dev))/1e6:.2f} ms")
print(f"  vrs TIMECODE ts[:3]={vrs_ts[:3]}")

# ---------------------------------------------------------------- (b) timestamp match
def _boxed(r):
    """HOT3D emits a row per (stream,hand,ts) even when the hand is NOT visible in that camera
    -> the pixel fields are EMPTY. A non-empty box == hand visible in this stream."""
    return r["x_min[pixel]"] not in ("", None)

all_rows = [r for r in csv.DictReader(open(os.path.join(SEQ_DIR, "box2d_hands.csv")))
            if r["stream_id"] == RGB_STREAM]
box_rows = [r for r in all_rows if _boxed(r)]
print(f"\n  box2d rows on {RGB_STREAM}: {len(all_rows)} total, {len(box_rows)} with a box "
      f"({len(all_rows)-len(box_rows)} empty = hand not visible in RGB)")
# timestamp axis = every RGB frame that has at least one row (boxed or not)
box_ts = np.array(sorted({int(r["timestamp[ns]"]) for r in all_rows}), dtype=np.int64)
d = np.load(CONTACT, allow_pickle=True)
con_ts = d["timestamps_ns"].astype(np.int64)

print("\n=== (b) timestamp domains (after device->timecode conversion) ===")
print(f"  vrs(tc) n={len(vrs_ts):5d}  [{vrs_ts.min()} .. {vrs_ts.max()}]")
print(f"  box2d   n={len(box_ts):5d}  [{box_ts.min()} .. {box_ts.max()}]")
print(f"  contact n={len(con_ts):5d}  [{con_ts.min()} .. {con_ts.max()}]")
print(f"  |box2d ∩ vrs(tc)|   = {len(np.intersect1d(box_ts, vrs_ts))}  (exact)")
print(f"  |contact ∩ box2d|   = {len(np.intersect1d(con_ts, box_ts))}  (exact)")

# nearest-match VRS(timecode) -> box2d timestamps; report residual
j = np.searchsorted(box_ts, vrs_ts).clip(1, len(box_ts) - 1)
best = np.where(np.abs(box_ts[j] - vrs_ts) < np.abs(box_ts[j - 1] - vrs_ts), j, j - 1)
resid = np.abs(box_ts[best] - vrs_ts)
print(f"  nearest-match residual: max={resid.max()} ns, median={np.median(resid):.0f} ns")
print(f"  frames matched within 1us: {(resid < 1000).sum()}/{len(vrs_ts)}")
assert (resid < 1_000_000).all(), "some RGB frame has no box2d timestamp within 1ms!"

# ---------------------------------------------------------------- (c) orientation
# pick a frame where BOTH hands have a box and the right hand is in contact
by_ts = {}
for r in box_rows:
    by_ts.setdefault(int(r["timestamp[ns]"]), {})[int(r["hand_index"])] = r

rc = d["right_contact"]; rp = d["right_present"]
cand = [ts for ts in con_ts[(rp) & (rc.sum(1) > 20)] if ts in by_ts and 0 in by_ts[ts] and 1 in by_ts[ts]]
print(f"\n=== (c) orientation ===\n  {len(cand)} candidate frames (right hand present + >20 contact verts + both boxes)")
if not cand:
    cand = [ts for ts in con_ts if ts in by_ts]
picks = [cand[len(cand) // 4], cand[len(cand) // 2], cand[3 * len(cand) // 4]]

# annotation timestamp (timecode) -> VRS RGB frame index, via the nearest-match computed above
ts2idx = {int(box_ts[best[i]]): i for i in range(len(vrs_ts))}
HAND_COL = {0: (255, 60, 60), 1: (60, 255, 60)}   # 0=left blue-ish, 1=right green (BGR)
HAND_NAME = {0: "left(0)", 1: "right(1)"}

for k, ts in enumerate(picks):
    i = ts2idx[int(ts)]
    img, _ = provider.get_image_data_by_index(rgb, i)
    raw = img.to_numpy_array()                       # RAW VRS orientation
    raw_bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)

    # draw box2d boxes on the RAW frame, unmodified coords
    vis = raw_bgr.copy()
    for hi, r in sorted(by_ts[int(ts)].items()):
        x0, x1 = float(r["x_min[pixel]"]), float(r["x_max[pixel]"])
        y0, y1 = float(r["y_min[pixel]"]), float(r["y_max[pixel]"])
        vr = float(r["visibility_ratio[%]"])
        c = HAND_COL[hi]
        cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), c, 4)
        cv2.putText(vis, f"{HAND_NAME[hi]} vis={vr:.2f}", (int(x0), max(0, int(y0) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, c, 3)
    cv2.putText(vis, "RAW VRS orientation + box2d coords as-is", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
    cv2.imwrite(f"{OUT}/{SEQ}_{k}_raw_boxed.jpg", cv2.resize(vis, (704, 704)))

    # what "upright" looks like (Aria RGB is stored rotated 90° CW from upright)
    up = cv2.rotate(raw_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.putText(up, "rot90 CCW (no boxes)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
    cv2.imwrite(f"{OUT}/{SEQ}_{k}_rot90ccw.jpg", cv2.resize(up, (704, 704)))

    # (d) right-hand crop as the loader takes it, and the left hand flipped to look right
    H, W = raw_bgr.shape[:2]
    for hi, r in sorted(by_ts[int(ts)].items()):
        x0, x1 = float(r["x_min[pixel]"]), float(r["x_max[pixel]"])
        y0, y1 = float(r["y_min[pixel]"]), float(r["y_max[pixel]"])
        src = raw_bgr
        if hi == 0:                       # LEFT hand -> flip image horizontally, flip bbox x
            src = cv2.flip(raw_bgr, 1)
            x0, x1 = W - x1, W - x0
        pad = 0.25 * max(x1 - x0, y1 - y0)
        cx0, cy0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
        cx1, cy1 = int(min(W, x1 + pad)), int(min(H, y1 + pad))
        crop = src[cy0:cy1, cx0:cx1]
        if crop.size:
            crop = cv2.resize(crop, (256, 256))
            tag = "LEFT->flipped (should look like a RIGHT hand)" if hi == 0 else "RIGHT as-is"
            cv2.putText(crop, tag.split(" ")[0], (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imwrite(f"{OUT}/{SEQ}_{k}_crop_{HAND_NAME[hi]}.jpg", crop)

print(f"  picked ts={picks} -> wrote overlays to {OUT}")
print("\n=== done ===")
