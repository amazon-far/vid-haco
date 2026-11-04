"""HOT3D VRS -> HACO-style per-sequence frame trees.

Decodes each labeled sequence's Aria RGB stream (214-1, 1408x1408) to

    <OUT>/<seq>/%06d.jpg          upright frames (rot90 CCW from raw VRS)
    <OUT>/<seq>/frames_index.json timecode_ns per frame + geometry

so multidata_vjepa's neighbour-clip builder (numbered frames: the HOT3D reader's frame_digits=6,
'.jpg') can build a real T-frame clip ending at any labeled target frame.

WHY UPRIGHT: Aria stores RGB rotated 90deg CW from upright. box2d_hands.csv is in RAW VRS pixel
coords (verified). Every other training dataset is upright, so we rotate the image CCW here and rotate
the boxes to match in build_haco_hot3d.py (rot90ccw_box) -- keeping the hand crops in the same
domain as ARCTIC/H2O/etc.

WHY TIMECODE: VRS capture_timestamp_ns is DEVICE time; every HOT3D annotation file (box2d, masks,
dynamic_objects, and hence our contact npz) is in TIMECODE time. We convert per frame and store the
timecode, so downstream joins are a plain integer lookup. (Residual of the conversion vs box2d is
<=1ns; we nearest-match with a 1ms tolerance.)
"""
import os, sys, json, glob, argparse, traceback
os.environ.setdefault("XR_LOG_LEVEL", "error")

RGB_STREAM = "214-1"          # Aria camera-rgb. Quest3 has NO rgb camera (mono slam only).
DEFAULT_OUT = "data/HOT3D/data"


def extract_one(seq, seq_dir, out_root, quality=90, limit=None, force=False):
    """Decode one sequence. Returns (seq, n_frames, status)."""
    import numpy as np
    import cv2
    from projectaria_tools.core import data_provider
    from projectaria_tools.core.stream_id import StreamId

    out_dir = os.path.join(out_root, seq)
    done_marker = os.path.join(out_dir, ".done")
    if os.path.isfile(done_marker) and not force:
        try:
            n = json.load(open(os.path.join(out_dir, "frames_index.json")))["n_frames"]
            return (seq, n, "cached")
        except Exception:
            pass                       # fall through and redo
    os.makedirs(out_dir, exist_ok=True)

    vrs = os.path.join(seq_dir, "recording.vrs")
    provider = data_provider.create_vrs_data_provider(vrs)
    sid = StreamId(RGB_STREAM)
    n = provider.get_num_data(sid)
    if limit:
        n = min(n, limit)

    tcs = []
    src_w = src_h = None
    for i in range(n):
        img, rec = provider.get_image_data_by_index(sid, i)
        arr = img.to_numpy_array()                       # RAW VRS orientation, RGB
        if src_w is None:
            src_h, src_w = arr.shape[:2]
        up = cv2.rotate(arr, cv2.ROTATE_90_COUNTERCLOCKWISE)      # -> upright
        bgr = cv2.cvtColor(up, cv2.COLOR_RGB2BGR)                 # cv2.imwrite wants BGR
        cv2.imwrite(os.path.join(out_dir, "%06d.jpg" % i), bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        tcs.append(int(provider.convert_from_device_time_to_timecode_ns(
            int(rec.capture_timestamp_ns))))

    index = dict(seq=seq, stream=RGB_STREAM, n_frames=n,
                 src_width=int(src_w), src_height=int(src_h),      # RAW dims (box2d coord space)
                 width=int(src_h), height=int(src_w),              # upright dims (after rot90)
                 rotation="rot90ccw", timecode_ns=tcs)
    with open(os.path.join(out_dir, "frames_index.json"), "w") as f:
        json.dump(index, f)
    open(done_marker, "w").close()
    return (seq, n, "ok")


def _worker(a):
    try:
        return extract_one(*a)
    except Exception as e:
        traceback.print_exc()
        return (a[0], 0, f"ERROR: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headset", default="aria", choices=["aria", "quest"])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--limit-frames", type=int, default=None, help="debug: only N frames/seq")
    ap.add_argument("--seqs", nargs="*", default=None, help="explicit seq names (default: all labeled)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.headset != "aria":
        sys.exit("Quest3 has no RGB camera (mono SLAM only) -- aria is the RGB source. "
                 "Grayscale Quest support is deliberately not wired up.")

    labeled = sorted(os.path.basename(p)[:-4]
                     for p in glob.glob(f"data/hot3d_contact/contact_data/{args.headset}/*.npz"))
    seqs = args.seqs if args.seqs else labeled
    root = f"data/hot3d_dataset/{args.headset}"
    todo = [(s, os.path.join(root, s), args.out, args.quality, args.limit_frames, args.force)
            for s in seqs if os.path.isfile(os.path.join(root, s, "recording.vrs"))]
    os.makedirs(args.out, exist_ok=True)
    print(f"[extract] {len(todo)} sequences -> {args.out}  ({args.workers} workers, q={args.quality})",
          flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    done = tot = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, a) for a in todo]
        for f in as_completed(futs):
            seq, n, st = f.result()
            done += 1
            tot += n
            print(f"[{done}/{len(todo)}] {seq:22s} {n:5d} frames  {st}", flush=True)
    print(f"[extract] DONE: {done} seqs, {tot} frames", flush=True)


if __name__ == "__main__":
    main()
