#!/usr/bin/env python3
"""
Cut a full-video HaWoR track down to the frame interval Stage A covers.

Why this exists: the tracker runs over the WHOLE video (SLAM needs the full sequence to solve
camera motion), so it returns one row per source frame. Stage A currently keeps every frame too, but
the combiner requires all three inputs to describe the SAME interval, exactly: it rejects a HaWoR
archive whose row count differs from the requested interval. So we slice the track to Stage A's range
before combining, which keeps that guarantee whatever range Stage A reports.

Per-frame arrays are sliced; per-sequence constants (faces, intrinsics, image size, provenance
strings and the whole `cop_viz_filter_*` scalar block) are carried through untouched. The
output's `frame_idx` is re-stamped to the ABSOLUTE interval start..end, which the combiner
accepts as-is and which keeps every archive in the run in one shared frame space.
"""
import argparse
import os
import sys

import numpy as np

# Arrays that are per-sequence even though their first axis could coincidentally equal the
# frame count: mesh topology, camera intrinsics, image dimensions. Everything else with a
# leading axis of length N is per-frame.
WHOLE_KEYS = {'faces', 'K', 'img_wh'}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--npz', required=True, help='HaWoR track (raw or filtered), one row per source frame')
    ap.add_argument('--out', required=True)
    ap.add_argument('--start-frame', type=int, required=True, help='inclusive absolute first frame')
    ap.add_argument('--end-frame', type=int, required=True, help='inclusive absolute last frame')
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    if 'frame_idx' not in z.files:
        raise SystemExit(f"[slice-track] {args.npz} has no frame_idx")
    fidx = np.asarray(z['frame_idx']).reshape(-1)
    n_total = len(fidx)
    s, e = int(args.start_frame), int(args.end_frame)
    if e < s:
        raise SystemExit(f"[slice-track] empty interval {s}..{e}")

    # The tracker indexes frames from 0 of the prepared video, so absolute frame == row index.
    # Verify that before trusting it, rather than assuming.
    if not np.array_equal(fidx, np.arange(n_total)):
        if int(fidx[0]) == s and len(fidx) == e - s + 1:
            print(f"[slice-track] already sliced to {s}..{e}; copying through", flush=True)
            lo, hi = 0, len(fidx)
        else:
            raise SystemExit(
                f"[slice-track] expected frame_idx 0..{n_total - 1}, got "
                f"{int(fidx[0])}..{int(fidx[-1])} — the track is not in the prepared video's frame space")
    else:
        if e > n_total - 1:
            raise SystemExit(f"[slice-track] interval {s}..{e} exceeds the track's {n_total} frames")
        lo, hi = s, e + 1

    n = hi - lo
    out = {}
    sliced, carried = [], []
    for k in z.files:
        a = z[k]
        if k == 'frame_idx':
            out[k] = np.arange(s, e + 1, dtype=np.int64)
            continue
        if k not in WHOLE_KEYS and a.ndim >= 1 and a.shape[0] == n_total:
            out[k] = a[lo:hi]
            sliced.append(k)
        else:
            out[k] = a
            carried.append(k)

    out['sliced_from_npz'] = np.asarray(os.path.abspath(args.npz))
    out['sliced_interval'] = np.asarray([s, e], dtype=np.int64)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"[slice-track] {n_total} -> {n} rows ({s}..{e}); sliced {len(sliced)} arrays, "
          f"carried {len(carried)} constants -> {args.out}", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
