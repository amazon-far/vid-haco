"""HOT3D contact labels + box2d -> per-sample preprocessed_data / splits (HACO-style layout).

Joins, per Aria sequence, keyed by TIMECODE ns (exact integer):
    contact npz      {left,right}_contact (T,778) uint8 + {left,right}_present
    box2d_hands.csv  per-hand pixel box on the RGB stream (RAW VRS coords) + visibility_ratio
    masks/*.csv      HOT3D's own per-frame quality masks (good exposure)
    frames_index.json  timecode -> extracted upright frame index  (from extract_frames.py)

Emits the layout loaders/HOT3D/dataset.py reads (the per-sample layout of HACO's preprocessed data):
    HOT3D/preprocessed_data/train/annot_data/<sid>.npz     bbox_hand_r  (xywh, upright coords)
    HOT3D/preprocessed_data/train/contact_data/<sid>.npy   (778,) uint8 contact @1cm
    HOT3D/splits/train.json                                ["<seq>/<hand>/<frame_idx>", ...]
    HOT3D/dataset_stats.json

sid = "<seq>-<hand>-<frame_idx:06d>"     e.g. P0001_10a27bf7-right-001234

GEOMETRY
  box2d is in RAW VRS pixel coords; we extract frames UPRIGHT (rot90 CCW), so boxes are rotated:
      (x0,y0,x1,y1) -> (y0, W-1-x1, y1, W-1-x0)        [W = raw width]
  then expanded to a SQUARE box with context (the crop stretches bbox_w x bbox_h onto the square
  network input, so a square box avoids anisotropic distortion; the other datasets ship
  roughly-square boxes with context, e.g. H2O hand boxes are ~334x346 px).

LEFT HANDS
  The crop is right-hand-centric. We emit left hands as PROX-style mirrored samples: the loader
  horizontally flips the whole frame (turning the left hand into a right-hand-looking crop), so the
  bbox is stored PRE-FLIPPED in x. The 778 contact vector is used as-is, because MANO left and
  right share vertex ordering.
"""
import os, sys, csv, json, glob, argparse
import numpy as np

HOT3D_ROOT = "data/hot3d_dataset"
CONTACT_ROOT = "data/hot3d_contact/contact_data"
HACO_HOT3D = "data/HOT3D"
RGB_STREAM = "214-1"
HANDS = {0: ("left", "left_contact", "left_present"),
         1: ("right", "right_contact", "right_present")}


def rot90ccw_box(x0, y0, x1, y1, W):
    """RAW VRS bbox -> upright (rot90 CCW) bbox. W = RAW width. Verified in probe_upright.py."""
    return y0, W - 1 - x1, y1, W - 1 - x0


def load_box2d(seq_dir):
    """ts -> hand_idx -> (x0,y0,x1,y1,vis) in RAW coords. Rows with an EMPTY box mean the hand is
    not visible in this camera -> dropped."""
    out = {}
    with open(os.path.join(seq_dir, "box2d_hands.csv")) as f:
        for r in csv.DictReader(f):
            if r["stream_id"] != RGB_STREAM or r["x_min[pixel]"] == "":
                continue
            out.setdefault(int(r["timestamp[ns]"]), {})[int(r["hand_index"])] = (
                float(r["x_min[pixel]"]), float(r["y_min[pixel]"]),
                float(r["x_max[pixel]"]), float(r["y_max[pixel]"]),
                float(r["visibility_ratio[%]"]))
    return out


def load_mask(seq_dir, name):
    """ts -> bool, for the RGB stream only."""
    p = os.path.join(seq_dir, "masks", name)
    out = {}
    if not os.path.isfile(p):
        return out
    with open(p) as f:
        for r in csv.DictReader(f):
            if r["stream_id"] == RGB_STREAM:
                out[int(r["timestamp[ns]"])] = (r["mask"].strip().lower() == "true")
    return out


def build_seq(seq, args):
    seq_dir = os.path.join(HOT3D_ROOT, "aria", seq)
    fidx_p = os.path.join(HACO_HOT3D, "data", seq, "frames_index.json")
    if not os.path.isfile(fidx_p):
        return seq, [], {"status": "no_frames"}
    fidx = json.load(open(fidx_p))
    raw_W = fidx["src_width"]                 # box2d coord space
    up_W = fidx["width"]                      # upright frame width (for the left-hand x-flip)
    up_H = fidx["height"]
    frame_tc = np.asarray(fidx["timecode_ns"], dtype=np.int64)   # frame_idx -> timecode

    # timecode -> frame index (nearest; conversion residual is <=1ns, frames are 33ms apart)
    order = np.argsort(frame_tc)
    tc_sorted = frame_tc[order]

    def tc_to_frame(ts):
        j = int(np.searchsorted(tc_sorted, ts))
        best, bd = -1, None
        for k in (j - 1, j):
            if 0 <= k < len(tc_sorted):
                dd = abs(int(tc_sorted[k]) - int(ts))
                if bd is None or dd < bd:
                    bd, best = dd, int(order[k])
        return (best, bd) if bd is not None and bd <= args.ts_tol_ns else (-1, bd)

    boxes = load_box2d(seq_dir)
    expo = load_mask(seq_dir, "mask_good_exposure.csv")

    d = np.load(os.path.join(CONTACT_ROOT, "aria", f"{seq}.npz"), allow_pickle=True)
    con_ts = d["timestamps_ns"].astype(np.int64)

    annot_dir = os.path.join(HACO_HOT3D, "preprocessed_data", "train", "annot_data")
    con_dir = os.path.join(HACO_HOT3D, "preprocessed_data", "train", "contact_data")
    os.makedirs(annot_dir, exist_ok=True)
    os.makedirs(con_dir, exist_ok=True)

    split, npos, ntot = [], 0, 0
    drop = dict(no_frame=0, no_box=0, low_vis=0, bad_expo=0, not_present=0, too_small=0)

    for ci, ts in enumerate(con_ts):
        ts = int(ts)
        hb = boxes.get(ts)
        if hb is None:
            drop["no_box"] += 1
            continue
        if args.require_exposure and not expo.get(ts, True):
            drop["bad_expo"] += 1
            continue
        fi, _ = tc_to_frame(ts)
        if fi < 0:
            drop["no_frame"] += 1
            continue

        for hi, (hname, ckey, pkey) in HANDS.items():
            if args.hands != "both" and hname != args.hands:
                continue
            if not bool(d[pkey][ci]):
                drop["not_present"] += 1
                continue
            if hi not in hb:
                drop["no_box"] += 1
                continue
            x0, y0, x1, y1, vis = hb[hi]
            if vis < args.vis_min:
                drop["low_vis"] += 1
                continue

            # RAW -> upright
            nx0, ny0, nx1, ny1 = rot90ccw_box(x0, y0, x1, y1, raw_W)
            w, h = nx1 - nx0, ny1 - ny0
            if min(w, h) < args.min_box:
                drop["too_small"] += 1
                continue

            # square + context expansion (isotropic; no stretch onto the 256x256 patch)
            cx, cy = nx0 + w / 2.0, ny0 + h / 2.0
            s = max(w, h) * args.expand
            bx, by = cx - s / 2.0, cy - s / 2.0

            # LEFT hand -> loader will cv2.flip(img,1); store the bbox pre-flipped in x
            if hi == 0:
                bx = up_W - bx - s

            sid = f"{seq}-{hname}-{fi:06d}"
            np.savez(os.path.join(annot_dir, sid + ".npz"),
                     bbox_hand_r=np.asarray([bx, by, s, s], np.float32),
                     visibility=np.float32(vis), timecode_ns=np.int64(ts),
                     hand=hname, img_wh=np.asarray([up_W, up_H], np.int32))
            contact = np.asarray(d[ckey][ci], np.uint8)
            np.save(os.path.join(con_dir, sid + ".npy"), contact)

            split.append(f"{seq}/{hname}/{fi}")
            ntot += 1
            npos += int(contact.sum() > 0)

    return seq, split, dict(status="ok", n=ntot, n_contact_pos=npos, drop=drop,
                            n_frames=int(fidx["n_frames"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", default="both", choices=["both", "left", "right"])
    ap.add_argument("--vis-min", type=float, default=0.5,
                    help="min HOT3D visibility_ratio for the hand in the RGB frame")
    ap.add_argument("--expand", type=float, default=1.3, help="square-bbox context expansion")
    ap.add_argument("--min-box", type=float, default=40.0, help="min tight-box side in px")
    ap.add_argument("--ts-tol-ns", type=int, default=1_000_000, help="timecode match tolerance (1ms)")
    ap.add_argument("--require-exposure", action="store_true", default=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seqs", nargs="*", default=None)
    args = ap.parse_args()

    seqs = args.seqs or sorted(
        os.path.basename(p)[:-4] for p in glob.glob(os.path.join(CONTACT_ROOT, "aria", "*.npz")))
    seqs = [s for s in seqs if os.path.isfile(os.path.join(HACO_HOT3D, "data", s, "frames_index.json"))]
    print(f"[build] {len(seqs)} sequences with extracted frames", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    all_split, stats, agg = [], {}, dict(n=0, n_contact_pos=0)
    drop_tot = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(build_seq, s, args): s for s in seqs}
        for k, f in enumerate(as_completed(futs), 1):
            seq, split, st = f.result()
            all_split.extend(split)
            stats[seq] = st
            if st["status"] == "ok":
                agg["n"] += st["n"]
                agg["n_contact_pos"] += st["n_contact_pos"]
                for kk, vv in st["drop"].items():
                    drop_tot[kk] = drop_tot.get(kk, 0) + vv
            if k % 20 == 0 or k == len(seqs):
                print(f"  [{k}/{len(seqs)}] {seq:22s} +{st.get('n',0):5d}  (total {agg['n']})", flush=True)

    all_split.sort()
    os.makedirs(os.path.join(HACO_HOT3D, "splits"), exist_ok=True)
    with open(os.path.join(HACO_HOT3D, "splits", "train.json"), "w") as f:
        json.dump(all_split, f)

    summary = dict(n_samples=agg["n"], n_sequences=len(seqs),
                   n_contact_positive=agg["n_contact_pos"],
                   contact_positive_rate=(agg["n_contact_pos"] / max(agg["n"], 1)),
                   dropped=drop_tot,
                   config=dict(hands=args.hands, vis_min=args.vis_min, expand=args.expand,
                               min_box=args.min_box, stream=RGB_STREAM, headset="aria"),
                   per_seq=stats)
    with open(os.path.join(HACO_HOT3D, "dataset_stats.json"), "w") as f:
        json.dump(summary, f, indent=1)

    print(f"\n[build] samples      : {agg['n']}")
    print(f"[build] contact>0    : {agg['n_contact_pos']} ({100*summary['contact_positive_rate']:.1f}%)")
    print(f"[build] dropped      : {drop_tot}")
    print(f"[build] splits/train.json + dataset_stats.json written under {HACO_HOT3D}")


if __name__ == "__main__":
    main()
