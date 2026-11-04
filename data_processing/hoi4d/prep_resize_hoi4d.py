#!/usr/bin/env python3
"""
One-time pre-resize of HOI4D align_rgb frames to a parallel half-resolution tree, so the
live V-JEPA training dataloader decodes small JPEGs instead of 1920x1080 ones (the decode
was the CPU bottleneck). The crop math in loaders/HOI4D/dataset_vjepa.py scales the bbox/affine
by the same factor, so the final 256x256 hand crop fed to the model is near-identical
(measured: mean-abs 1.6/255, PSNR 38.6 dB, SSIM 0.995 at f=0.5 vs full-res).

Mapping (trivial): for every
  .../HOI4D_color/HOI4D_release/<seq...>/align_rgb/<NNNNN>.jpg
write
  .../HOI4D_color_half/HOI4D_release/<seq...>/align_rgb/<NNNNN>.jpg
resized by FACTOR with cv2.INTER_AREA (the correct downscale filter) at JPEG quality QUALITY.

Idempotent: skips a target that already exists with non-zero size, so it is safe to resume.
Writes next to the source tree.

Usage:
  python data_processing/hoi4d/prep_resize_hoi4d.py [--factor 0.5] [--quality 95] [--workers N] [--limit N]
"""
import os
import sys
import time
import argparse
import multiprocessing as mp

import cv2

# resolve the real (local-SSD) color tree via the repo's data symlink
SRC_COLOR = os.path.join('data', 'HOI4D', 'data', 'HOI4D_color')
SRC_TAG = 'HOI4D_color'
DST_TAG = 'HOI4D_color_half'

# pin OpenCV to a single thread per worker so N workers != N*threads oversubscription
cv2.setNumThreads(1)


def list_jobs(src_release):
    """Yield (src_jpg, rel_path_under_release) for every align_rgb/*.jpg."""
    for root, dirs, files in os.walk(src_release):
        if os.path.basename(root) != 'align_rgb':
            continue
        for f in files:
            if f.endswith('.jpg'):
                src = os.path.join(root, f)
                rel = os.path.relpath(src, src_release)
                yield src, rel


def resize_one(args):
    src, dst, factor, quality = args
    try:
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            return (1, 0)  # skipped (already done)
        img = cv2.imread(src, cv2.IMREAD_COLOR)
        if img is None:
            return (0, 1)  # read fail
        h, w = img.shape[:2]
        nw, nh = int(round(w * factor)), int(round(h * factor))
        small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        d = os.path.dirname(dst)
        os.makedirs(d, exist_ok=True)
        # temp file MUST keep a .jpg extension: cv2.imwrite picks the encoder from the
        # extension, so a bare .tmpNNNN suffix fails with "could not find a writer".
        tmp = dst + '.%d.tmp.jpg' % os.getpid()
        ok = cv2.imwrite(tmp, small, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return (0, 1)
        os.replace(tmp, dst)  # atomic; partial files never appear as final
        return (0, 0)  # written
    except Exception as e:
        sys.stderr.write(f"ERR {src}: {e}\n")
        return (0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--factor', type=float, default=0.5)
    ap.add_argument('--quality', type=int, default=95)
    ap.add_argument('--workers', type=int, default=max(1, mp.cpu_count()))
    ap.add_argument('--limit', type=int, default=0, help='if >0, only process this many (debug)')
    args = ap.parse_args()

    src_color = os.path.realpath(SRC_COLOR)
    assert SRC_TAG in src_color, src_color
    dst_color = src_color.replace(SRC_TAG, DST_TAG)
    src_release = os.path.join(src_color, 'HOI4D_release')
    dst_release = os.path.join(dst_color, 'HOI4D_release')

    print(f"[prep] src  = {src_color}")
    print(f"[prep] dst  = {dst_color}")
    print(f"[prep] factor={args.factor} quality={args.quality} workers={args.workers}", flush=True)

    print("[prep] scanning for jpgs ...", flush=True)
    jobs = []
    for src, rel in list_jobs(src_release):
        dst = os.path.join(dst_release, rel)
        jobs.append((src, dst, args.factor, args.quality))
        if args.limit and len(jobs) >= args.limit:
            break
    total = len(jobs)
    print(f"[prep] {total} frames to (re)size", flush=True)

    t0 = time.time()
    written = skipped = failed = 0
    with mp.Pool(args.workers) as pool:
        for i, (sk, fl) in enumerate(pool.imap_unordered(resize_one, jobs, chunksize=64)):
            skipped += sk
            failed += fl
            if sk == 0 and fl == 0:
                written += 1
            if (i + 1) % 20000 == 0:
                el = time.time() - t0
                rate = (i + 1) / max(1e-9, el)
                eta = (total - (i + 1)) / max(1e-9, rate)
                print(f"[prep] {i+1}/{total} done  written={written} skipped={skipped} "
                      f"failed={failed}  {rate:.0f} img/s  ETA {eta/60:.1f} min", flush=True)

    el = time.time() - t0
    print(f"[prep] DONE total={total} written={written} skipped={skipped} failed={failed} "
          f"in {el:.1f}s ({el/60:.2f} min)  avg {total/max(1e-9,el):.0f} img/s", flush=True)
    print(f"[prep] resized tree at: {dst_color}", flush=True)


if __name__ == '__main__':
    main()
