"""Build the ARCTIC_allo dataset index (allocentric cams 1-8).

DESIGN
  * SEPARATE dataset -- ARCTIC (ego, cam 0) is NOT touched, so the 10-dataset baseline stays
    reproducible and the existing preprocessed_data/splits are untouched.
  * IMAGES are shared: data/ARCTIC/data/cropped_images/<subj>/<seq>/<cam>/%05d.jpg  (cams 1..8).
  * CONTACT is shared and VIEW-INDEPENDENT: the 778-vertex contact map is a property of the 3D
    hand-object geometry, so the SAME label applies to every camera. We reuse ARCTIC's existing
    contact_data/<subj>-<seq>-0-<frame>.npy -- no duplication of 1.5M files.
  * BBOX is the only per-view quantity. Derived from ARCTIC's crop geometry (crop_coords.py; validated:
    the 21 joints land on the hand in every view, and the 2.0x expansion matches HACO's cam-0 scale
    w=2.015 h=1.977):
        j2d  = data_dict[seq]['2d']['joints.right'][row, cam]      # ORIGINAL image coords
        k    = to_crop_coords(j2d, data_dict[seq]['bbox'][row, cam])  # -> 1000x1000 cropped coords
        bbox = tight(k) expanded 2.0x
  * Stored as ONE index.npz (arrays), not 1.5M per-sample npz -- O(1) lookup, ~35 MB.

QUALITY FILTER (makes ARCTIC_allo cleaner than ARCTIC itself, where 23% of shipped cam-0 boxes have
their centre off-image and widths reach 70,746 px):
  keep only boxes whose centre is inside the 1000x1000 image and whose size is sane.
"""
import numpy as np
import json, os, re, glob, argparse

from crop_coords import CROP_DIM, to_crop_coords

ARCTIC = 'data/ARCTIC'
OUT = 'data/ARCTIC_allo'
IMROOT = os.path.join(ARCTIC, 'data', 'cropped_images')
CONTACT = os.path.join(ARCTIC, 'preprocessed_data', 'train', 'contact_data')
DIM = CROP_DIM
EXPAND = 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cams', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8])
    ap.add_argument('--min-box', type=float, default=30.0)
    ap.add_argument('--max-box', type=float, default=900.0)
    args = ap.parse_args()

    print('loading p1_train.npy ...', flush=True)
    p1 = np.load('data/arctic_allo_work/p1_train.npy', allow_pickle=True).item()
    dd = p1['data_dict']

    # frames that actually HAVE a contact label (from ARCTIC's cam-0 preprocessed set)
    have = {}
    for f in glob.glob(os.path.join(CONTACT, '*.npy')):
        b = os.path.basename(f)[:-4]                       # <subj>-<seq>-0-<frame>
        m = re.match(r'(s\d+)-(.+)-0-(\d+)$', b)
        if m:
            have.setdefault(f'{m.group(1)}/{m.group(2)}', set()).add(int(m.group(3)))
    print(f'  sequences with contact labels: {len(have)}   '
          f'labeled frames: {sum(len(v) for v in have.values())}', flush=True)

    seqs, seq_ids, cams, frames, bboxes = [], [], [], [], []
    drop = dict(no_img=0, off_image=0, bad_size=0, nan=0)
    kept_by_cam = {c: 0 for c in args.cams}

    for si, (seq, labeled) in enumerate(sorted(have.items())):
        if seq not in dd:
            continue
        e = dd[seq]
        j2d, bb = e['2d']['joints.right'], e['bbox']
        # frame -> row mapping is the IDENTITY: row = frame.
        # Verified against HACO's shipped cam-0 bbox_hand_r: with row=frame the reconstruction is
        # EXACT (centre distance 0.0000 px, expansion exactly 2.0000x, std 0). With row=frame-10 the
        # centre is off by a median of 65 px (220 px while the hand is moving) -- i.e. the box lands
        # where the hand WAS 10 frames earlier.
        seqs.append(seq)
        subj, sq = seq.split('/')
        for cam in args.cams:
            cam_dir = os.path.join(IMROOT, subj, sq, str(cam))
            if not os.path.isdir(cam_dir):
                drop['no_img'] += len(labeled)
                continue
            present = {int(x[:-4]) for x in os.listdir(cam_dir) if x.endswith('.jpg')}
            for fr in sorted(labeled):
                if fr not in present:
                    drop['no_img'] += 1
                    continue
                r = fr                       # row == frame (verified exact vs HACO cam-0)
                if r < 0 or r >= j2d.shape[0]:
                    drop['no_img'] += 1
                    continue
                k = to_crop_coords(j2d[r, cam], bb[r, cam])
                if not np.isfinite(k).all():
                    drop['nan'] += 1
                    continue
                x0, y0 = k[:, 0].min(), k[:, 1].min()
                x1, y1 = k[:, 0].max(), k[:, 1].max()
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                w, h = (x1 - x0) * EXPAND, (y1 - y0) * EXPAND
                if not (0 <= cx <= DIM and 0 <= cy <= DIM):
                    drop['off_image'] += 1
                    continue
                if not (args.min_box <= max(w, h) <= args.max_box):
                    drop['bad_size'] += 1
                    continue
                seq_ids.append(si); cams.append(cam); frames.append(fr)
                bboxes.append([cx - w / 2, cy - h / 2, w, h])
                kept_by_cam[cam] += 1
        if (si + 1) % 40 == 0:
            print(f'  [{si+1}/{len(have)}] {seq}  kept so far: {len(frames)}', flush=True)

    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, 'index.npz'),
             seqs=np.array(seqs, dtype=object),
             seq_ids=np.asarray(seq_ids, np.int16),
             cams=np.asarray(cams, np.uint8),
             frames=np.asarray(frames, np.int32),
             bbox=np.asarray(bboxes, np.float32))
    stats = dict(n_samples=len(frames), n_sequences=len(seqs), cams=args.cams,
                 kept_by_cam=kept_by_cam, dropped=drop, expand=EXPAND, dim=DIM)
    with open(os.path.join(OUT, 'dataset_stats.json'), 'w') as f:
        json.dump(stats, f, indent=1)

    print(f"\n[allo] samples : {len(frames)}")
    print(f"[allo] per cam : {kept_by_cam}")
    print(f"[allo] dropped : {drop}")
    print(f"[allo] -> {OUT}/index.npz")


if __name__ == '__main__':
    main()
