"""
Slice a Stage-A meta npz (frame_idx, bbox_sm, crop_white, fps, out_shape) into K contiguous shards.

Each shard file is itself a valid Stage-A meta over a sub-range of the segment. Because Stage B / W /
render read the FULL video and index it by the absolute `frame_idx` stored here, running a shard's
sub-meta against the full video is bit-identical to the unsharded run -- Stage B's clip history
(t-i*stride) reaches back across the shard boundary into the previous shard's frames automatically.

fps is overridden (--fps) so the rendered playback speed is what the user wants, independent of the
container's tag.
"""
import os, argparse, numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--prefix', required=True)      # e.g. IMG_3970
    ap.add_argument('--num_shards', type=int, required=True)
    ap.add_argument('--fps', type=float, default=0.0, help='>0 override output fps; 0 keep Stage-A fps')
    ap.add_argument('--manifest', default='', help='optional path to write shard list (one path per line)')
    ap.add_argument('--wilor', default='', help='optional cached full-video stageW_wilor.npz to slice per '
                                                'shard -> outdir/shard{k}_wilor.npz (skips per-shard WiLoR recompute)')
    ap.add_argument('--start-frame', type=int, default=None,
                    help='optional inclusive absolute source-frame start')
    ap.add_argument('--end-frame', type=int, default=None,
                    help='optional inclusive absolute source-frame end')
    args = ap.parse_args()

    if (args.start_frame is not None and args.end_frame is not None
            and args.start_frame > args.end_frame):
        raise SystemExit(f'[slice] invalid frame range: start {args.start_frame} '
                         f'is after end {args.end_frame}')

    m = np.load(args.meta)
    full_fi = m['frame_idx']
    if full_fi.ndim != 1:
        raise SystemExit(f'[slice] frame_idx must be 1-D, got shape {full_fi.shape}')
    if not np.issubdtype(full_fi.dtype, np.number) or not np.all(np.isfinite(full_fi)):
        raise SystemExit('[slice] frame_idx must contain finite numeric values')
    if not np.all(full_fi == np.floor(full_fi)):
        raise SystemExit('[slice] frame_idx must contain integer-valued frame numbers')

    select = np.ones(len(full_fi), dtype=bool)
    if args.start_frame is not None:
        select &= full_fi >= args.start_frame
    if args.end_frame is not None:
        select &= full_fi <= args.end_frame
    fi = full_fi[select]
    N = len(fi)
    if N == 0:
        raise SystemExit('[slice] requested frame range is empty')
    if N > 1 and not np.all(np.diff(fi) == 1):
        raise SystemExit('[slice] requested frame range is not contiguous')
    if args.start_frame is not None and int(fi[0]) != args.start_frame:
        raise SystemExit(f'[slice] start frame {args.start_frame} is unavailable')
    if args.end_frame is not None and int(fi[-1]) != args.end_frame:
        raise SystemExit(f'[slice] end frame {args.end_frame} is unavailable')

    # Compressed NpzFile members are decompressed on every lookup. Materialize each
    # large per-frame array and apply the range selection once before sharding.
    bbox_sm = m['bbox_sm']
    crop_white = m['crop_white']
    for name, value in (('bbox_sm', bbox_sm), ('crop_white', crop_white)):
        if value.ndim < 1 or value.shape[0] != len(full_fi):
            raise SystemExit(f'[slice] {name} first dimension must match frame_idx '
                             f'({value.shape if value.ndim else "scalar"} vs {len(full_fi)})')
    bbox_sm = bbox_sm[select]
    crop_white = crop_white[select]

    # Optional: cached full-video WiLoR to slice alongside the meta (same [s:e] ranges).
    # Explicit --wilor wins; otherwise auto-detect a pre-placed <outdir>/_full_wilor.npz so an
    # UNMODIFIED driver picks up the reuse just by having the cache staged into the work dir.
    wpath = args.wilor or os.path.join(args.outdir, '_full_wilor.npz')
    W = None
    if wpath and os.path.exists(wpath):
        W = dict(np.load(wpath, allow_pickle=True))
        wfi = W.get('frame_idx')
        if wfi is not None and np.array_equal(wfi, full_fi):
            W = {
                key: (value[select] if hasattr(value, 'shape') and value.ndim >= 1
                      and value.shape[0] == len(full_fi) else value)
                for key, value in W.items()
            }
            wfi = W['frame_idx']
        if wfi is None or len(wfi) != N or not np.array_equal(wfi, fi):
            raise SystemExit(f'[slice] wilor frame_idx mismatch (W has {0 if wfi is None else len(wfi)}, '
                             f'meta has {N}); cannot reuse')
        print(f'[slice] reusing cached WiLoR {wpath} ({N} frames) -> per-shard slices', flush=True)
    K = max(1, min(args.num_shards, N))
    fps = np.float32(args.fps) if args.fps > 0 else m['fps']
    os.makedirs(args.outdir, exist_ok=True)
    bounds = np.linspace(0, N, K + 1).astype(int)

    wh_kw = {'img_wh': m['img_wh']} if 'img_wh' in m.files else {}   # carry video WxH through to each shard
    written = []
    for k in range(K):
        s, e = int(bounds[k]), int(bounds[k + 1])
        if e <= s:
            continue
        out = os.path.join(args.outdir, f'{args.prefix}_shard{k:02d}_meta.npz')
        np.savez_compressed(
            out,
            frame_idx=fi[s:e], bbox_sm=bbox_sm[s:e],
            crop_white=crop_white[s:e],
            fps=fps, out_shape=m['out_shape'], **wh_kw,
        )
        written.append(out)
        print(f'shard{k:02d}: pos[{s}:{e}] N={e - s} abs[{int(fi[s])}..{int(fi[e - 1])}] -> {out}', flush=True)

        # Slice the cached WiLoR the same way (per-frame arrays [s:e]; shared arrays whole).
        if W is not None:
            wout = os.path.join(args.outdir, f'shard{k:02d}_wilor.npz')
            wslice = {kk: (vv[s:e] if hasattr(vv, 'shape') and vv.ndim >= 1 and vv.shape[0] == N else vv)
                      for kk, vv in W.items()}
            np.savez_compressed(wout, **wslice)
            print(f'          wilor -> {wout}', flush=True)

    print(f'[slice] {args.prefix}: N={N} abs[{int(fi[0])}..{int(fi[-1])}] '
          f'fps={float(fps):.3f} -> {len(written)} shards', flush=True)
    if args.manifest:
        with open(args.manifest, 'w') as f:
            f.write('\n'.join(written) + '\n')


if __name__ == '__main__':
    main()
