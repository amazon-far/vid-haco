#!/usr/bin/env python3
"""
haco_contact_infer — video in, mesh-top hand-contact video out.

    ./run.py --video clip.mp4 --out-dir out/clip --model vitl

Runs the whole chain and parallelises everything that can overlap:

    prepare      normalise to constant frame rate (mirror if the hand is left)
      |
    Stage A      hand detection (ONNX detector) + smoothed bbox
      |
      +--------------------------+                    <-- these two run CONCURRENTLY
      |                          |
    HaWoR tracking            Stage B  contact inference, sharded over the other GPUs
    (1 GPU, SLAM; tracks the     |
     hands Stage A detected)     |
      |                          |
    Butterworth filter <---------+
      |
    combine  (contact + filtered track -> one npz)
      |
    render   mesh-top visualisation, sharded over all GPUs, concatenated

The two selectable contact models are the ViT-L 12-dataset epoch 6 checkpoint (`--model vitl`)
and the ViT-H 8-frame-sequence epoch 1 checkpoint (`--model vith`); they use different Stage-B
runners and input resolutions, which this script handles. Paths to environments, the tracker and
the checkpoints live in config.env.

Stages are checkpointed: re-run with --resume to skip whatever already completed, which makes
re-rendering at a new threshold or trying the other model cheap (the tracking and the filter are
model-independent and are reused as-is).
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = ROOT / 'stages'

# Which Stage-B runner and backbone each model needs. The ViT-L is a V-JEPA 2.1 large at 384px
# built from the vjepa2 repo cloned into third_party/ (backbone weights come from the checkpoint); the ViT-H is a
# transformers VJEPA2Model at 256px, so it needs facebook/vjepa2-vith-fpc64-256 in the HF cache.
MODELS = {
    'vitl': {
        'ckpt_key': 'CKPT_VITL',
        'script': 'stageB_vitl.py',
        'model_id': 'vjepa2.1-vitl-384',
        'img_size': 384,
        'batch': 16,
        'needs_hf_cache': False,
        'label': 'V-JEPA 2.1 ViT-L/384 + DPT (12-dataset, epoch 6)',
    },
    'vith': {
        'ckpt_key': 'CKPT_VITH',
        'script': 'stageB_vith.py',
        'model_id': 'facebook/vjepa2-vith-fpc64-256',
        'img_size': None,          # this runner has no --img_size; it is fixed at 256
        'batch': 8,
        'needs_hf_cache': True,
        'label': 'V-JEPA 2 ViT-H/256 + DPT (8-frame seq, epoch 1)',
    },
}


# --------------------------------------------------------------------------- config
def load_config(path, overrides):
    """Read config.env (KEY=VALUE, # comments, ${VAR} references to earlier keys)."""
    cfg = {}
    if path.is_file():
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            k, v = k.strip(), v.strip()
            cfg[k] = re.sub(r'\$\{(\w+)\}', lambda m: cfg.get(m.group(1), os.environ.get(m.group(1), '')), v)
    # Precedence: config.env < environment < -e overrides, so a one-off run never needs an edit.
    for k in list(cfg):
        if k in os.environ:
            cfg[k] = os.environ[k]
    for item in overrides:
        if '=' not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        k, v = item.split('=', 1)
        cfg[k.strip()] = v.strip()
    return cfg


def base_env(cfg, gpu=None):
    """Environment shared by every haco_render subprocess."""
    env = dict(os.environ)
    env.update(
        PYTHONPATH=f"{ROOT}:{STAGES}",
        PYOPENGL_PLATFORM='egl',
        EGL_PLATFORM='surfaceless',
        LD_LIBRARY_PATH=f"{cfg.get('HACO_ENV_LIB', '')}:{os.environ.get('LD_LIBRARY_PATH', '')}",
        VJEPA2_REPO=str(ROOT / 'third_party' / 'vjepa2_repo'),
        HF_HOME=cfg.get('HF_CACHE', ''),
        HF_HUB_CACHE=cfg.get('HF_CACHE', ''),
        TORCH_HOME=cfg.get('TORCH_HOME', ''),
        TRANSFORMERS_OFFLINE='1',
        HF_HUB_OFFLINE='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        # Bound per-process CPU threads: with up to 8 workers on a 192-core box the default
        # (one thread per core each) would oversubscribe badly.
        OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', MKL_NUM_THREADS='8', NUMEXPR_NUM_THREADS='8',
    )
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    return env


# --------------------------------------------------------------------------- process helpers
class Job:
    """One subprocess with its log file, so parallel stages stay debuggable after the fact."""

    def __init__(self, name, cmd, log, env, cwd=ROOT):
        self.name, self.cmd, self.log = name, [str(c) for c in cmd], Path(log)
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.log, 'w')
        self.fh.write(f"# {name}\n# {' '.join(shlex.quote(c) for c in self.cmd)}\n\n")
        self.fh.flush()
        self.t0 = time.time()
        self.proc = subprocess.Popen(self.cmd, stdout=self.fh, stderr=subprocess.STDOUT,
                                     env=env, cwd=str(cwd))

    def wait(self, check=True):
        rc = self.proc.wait()
        self.fh.close()
        dt = time.time() - self.t0
        status = 'ok' if rc == 0 else f"FAILED rc={rc}"
        print(f"    [{self.name}] {status} in {dt:.1f}s  ({self.log})", flush=True)
        if check and rc != 0:
            tail = self.log.read_text().splitlines()[-25:]
            print('\n'.join(f"      | {t}" for t in tail), file=sys.stderr)
            raise SystemExit(f"stage {self.name} failed — see {self.log}")
        return rc

    @property
    def done(self):
        return self.proc.poll() is not None


def run_sync(name, cmd, log, env, cwd=ROOT, check=True):
    return Job(name, cmd, log, env, cwd).wait(check=check)


def wait_all(jobs, check=True):
    bad = [j for j in jobs if j.wait(check=False) != 0]
    if bad and check:
        for j in bad:
            print('\n'.join(f"      | {t}" for t in j.log.read_text().splitlines()[-25:]), file=sys.stderr)
        raise SystemExit(f"{len(bad)} of {len(jobs)} parallel jobs failed: {[j.name for j in bad]}")
    return bad


def capture_json(log_path, tag):
    """Pull a `TAG {json}` line out of a stage log (how sub-stages report structured results)."""
    for line in Path(log_path).read_text().splitlines():
        if line.startswith(tag + ' '):
            return json.loads(line[len(tag) + 1:])
    raise SystemExit(f"{log_path}: no {tag} line — the stage did not report its result")


def split_ranges(n, k, minimum):
    """Split n rows into at most k contiguous chunks, each with at least `minimum` rows."""
    k = max(1, min(k, max(1, n // max(1, minimum))))
    edges = [round(i * n / k) for i in range(k + 1)]
    return [(edges[i], edges[i + 1]) for i in range(k) if edges[i + 1] > edges[i]]


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--video', required=True, help='input video (any container ffmpeg can read)')
    ap.add_argument('--out-dir', required=True, help='run directory; everything is written under it')
    ap.add_argument('--model', default='vitl', choices=sorted(MODELS),
                    help='contact checkpoint to run (default: vitl)')
    ap.add_argument('--ckpt', default='', help='override the checkpoint path for --model')
    ap.add_argument('--hand', default='auto', choices=['auto', 'right', 'left'],
                    help='which hand to track; left is mirrored (default: auto-detect)')
    ap.add_argument('--fps', default='', help="output constant frame rate, e.g. 30 or 30000/1001 "
                                              "(default: the source's nominal rate)")
    ap.add_argument('--start-frame', type=int, default=0, help='inclusive first source frame')
    ap.add_argument('--end-frame', type=int, default=-1, help='inclusive last source frame (-1 = end)')
    ap.add_argument('--thr', type=float, default=None, help='contact probability threshold')
    ap.add_argument('--mesh-alpha', type=float, default=None, help='AR overlay mesh opacity')
    ap.add_argument('--filter', default='butterworth', choices=['butterworth', 'none'],
                    help='hand-track smoothing (default: zero-phase 2 Hz Butterworth)')
    ap.add_argument('--cutoff-hz', type=float, default=2.0, help='Butterworth cutoff')
    ap.add_argument('--gpus', default='', help='comma-separated GPU ids (default: config.env GPUS)')
    ap.add_argument('--mem-frac', type=float, default=None, help='per-process GPU memory cap [0-1]')
    ap.add_argument('--contact-shards', type=int, default=0, help='Stage-B shards (default: one per GPU)')
    ap.add_argument('--render-shards', type=int, default=0, help='render shards (default: one per GPU)')
    ap.add_argument('--resume', action='store_true', help='skip stages that already completed')
    ap.add_argument('--skip-render', action='store_true', help='stop after the combined npz')
    ap.add_argument('--keep-work', action='store_true', help='keep intermediates (default: kept anyway)')
    ap.add_argument('--set', dest='overrides', action='append', default=[], metavar='KEY=VALUE',
                    help='override a config.env key')
    ap.add_argument('--dry-run', action='store_true', help='print the plan and exit')
    args = ap.parse_args()

    cfg = load_config(ROOT / 'config.env', args.overrides)
    spec = MODELS[args.model]
    ckpt = args.ckpt or cfg.get(spec['ckpt_key'], '')
    if ckpt and not Path(ckpt).is_absolute() and not Path(ckpt).is_file():
        ckpt = str(ROOT / ckpt)  # config.env paths are relative to the project root
    thr = args.thr if args.thr is not None else float(cfg.get('THR', 0.52))
    mesh_alpha = args.mesh_alpha if args.mesh_alpha is not None else float(cfg.get('MESH_ALPHA', 0.6))
    mem_frac = args.mem_frac if args.mem_frac is not None else float(cfg.get('MEM_FRAC', 0.35))
    clip_len, stride = int(cfg.get('CLIP_LEN', 8)), int(cfg.get('STRIDE', 2))
    gpus = [g.strip() for g in (args.gpus or cfg.get('GPUS', '0')).split(',') if g.strip() != '']
    if not gpus:
        raise SystemExit('no GPUs configured — set GPUS in config.env or pass --gpus')

    src_video = Path(args.video).resolve()
    out = Path(args.out_dir).resolve()
    work, logs = out / 'work', out / 'logs'
    for d in (out, work, logs):
        d.mkdir(parents=True, exist_ok=True)
    tag = f"{src_video.stem}_{args.model}"
    marks = work / '.done'
    marks.mkdir(exist_ok=True)

    # Resume markers. Preparing the video, tracking the hand and filtering the track do not depend
    # on which contact model runs, so a second run with the other --model reuses them; Stage B, the
    # combine and the render do, so their markers carry the model name. Without that split,
    # `--model vith --resume` after a vitl run would silently keep the vitl contact maps.
    MODEL_SCOPED = {'stageB', 'combine', 'render'}

    def _key(step):
        return f"{step}.{args.model}" if step in MODEL_SCOPED else step

    def done(step):
        return args.resume and (marks / _key(step)).exists()

    def mark(step):
        (marks / _key(step)).write_text(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))

    if not ckpt or not Path(ckpt).is_file():
        raise SystemExit(f"checkpoint for --model {args.model} not found: {ckpt!r}\n"
                         f"set {spec['ckpt_key']} in config.env or pass --ckpt")

    # HaWoR takes the last GPU and runs alongside Stage B, which uses the rest. With a single
    # GPU everything shares it (slower, still correct).
    gpu_track = gpus[-1]
    gpus_contact = gpus[:-1] or gpus

    print(f"=== haco_contact_infer ===")
    print(f"  video      {src_video}")
    print(f"  model      {args.model}  ({spec['label']})")
    print(f"  checkpoint {ckpt}")
    print(f"  out        {out}")
    print(f"  gpus       tracking={gpu_track}  contact={','.join(gpus_contact)}  render={','.join(gpus)}")
    print(f"  thr={thr}  mesh_alpha={mesh_alpha}  mem_frac={mem_frac}  clip_len={clip_len} stride={stride}")
    if args.dry_run:
        return 0

    haco_py = cfg['HACO_PY']
    hawor_py = cfg['HAWOR_PY']
    hawor_root = Path(cfg['HAWOR_ROOT'])
    t_start = time.time()

    # ---------------------------------------------------------------- 0. hand + prepare
    hand = args.hand
    if hand == 'auto':
        if done('probe'):
            hand = capture_json(logs / 'probe.log', 'PROBE_JSON')['hand']
            print(f"[0] hand (cached): {hand}")
        else:
            print('[0] probing which hand is in the video...')
            run_sync('probe', [haco_py, STAGES / 'probe_hand.py', '--video', src_video,
                               '--detector_path', ROOT / 'data/base_data/demo_data/wilor_detector.onnx'],
                     logs / 'probe.log', base_env(cfg, gpus[0]))
            hand = capture_json(logs / 'probe.log', 'PROBE_JSON')['hand']
            mark('probe')
            print(f"    detected hand: {hand}")

    prepared = work / 'input_cfr.mp4'
    if done('prepare') and prepared.is_file():
        prep = json.loads((work / 'prepare.json').read_text())
        print(f"[0] prepare (cached): {prep['n_frames']} frames @ {prep['fps']:.3f} fps")
    else:
        print(f"[0] preparing video (constant frame rate{', mirrored' if hand == 'left' else ''})...")
        cmd = [sys.executable, STAGES / 'prepare_video.py', '--video', src_video, '--out', prepared,
               '--hand', hand, '--fps', args.fps, '--start-frame', args.start_frame,
               '--end-frame', args.end_frame, '--report', work / 'prepare.json']
        run_sync('prepare', cmd, logs / 'prepare.log', base_env(cfg))
        prep = json.loads((work / 'prepare.json').read_text())
        mark('prepare')
    n_frames, fps = int(prep['n_frames']), float(prep['fps'])
    print(f"    prepared: {n_frames} frames, {prep['width']}x{prep['height']} @ {fps:.3f} fps")

    # ---------------------------------------------------------------- 1. Stage A detect
    stage_a = work / 'stageA_full.npz'
    if done('stageA') and stage_a.is_file():
        print('[1] Stage A detect (cached)')
    else:
        print(f"[1] Stage A: detecting the {hand if hand == 'right' else 'mirrored ' + hand} hand "
              f"on GPU {gpus_contact[0]}...")
        run_sync('stageA', [haco_py, STAGES / 'stageA_detect.py', '--video', prepared,
                            '--out', stage_a, '--detector_path',
                            ROOT / 'data/base_data/demo_data/wilor_detector.onnx',
                            # The video is already mirrored for a left hand, so the detector always
                            # looks for a RIGHT hand — that is what the model was trained on.
                            '--hand', 'right'],
                 logs / 'stageA.log', base_env(cfg, gpus_contact[0]))
        mark('stageA')

    # ---------------------------------------------------------------- 2. tracking (background)
    # HaWoR tracks the hand from Stage A's detections (it runs no detector of its own here).
    hawor_raw = work / 'hawor_raw.npz'
    track_job = None
    if done('hawor') and hawor_raw.is_file():
        print('[2] HaWoR tracking (cached)')
    else:
        print(f"[2] HaWoR tracking on GPU {gpu_track} (background, concurrent with Stage B)...")
        # The tracker's env is pinned to an older torch (1.13) whose caching allocator rejects
        # `expandable_segments`, so it gets a clean environment rather than the haco_render one.
        env = dict(os.environ)
        env.pop('PYTORCH_CUDA_ALLOC_CONF', None)
        env.update(CUDA_VISIBLE_DEVICES=str(gpu_track))
        track_job = Job('hawor', [hawor_py, hawor_root / 'run_hawor_headless.py',
                                  '--video-path', prepared, '--output', hawor_raw,
                                  '--detections', stage_a,
                                  '--source-fps', fps, '--expected-frames', n_frames],
                        logs / 'hawor.log', env, cwd=hawor_root)

    rng = subprocess.run([haco_py, '-c',
                          "import numpy as np,sys; z=np.load(sys.argv[1]); f=z['frame_idx']; "
                          "print(int(f.min()), int(f.max()), len(f))", str(stage_a)],
                         check=True, capture_output=True, text=True, env=base_env(cfg)).stdout.split()
    seg_start, seg_end, seg_n = int(rng[0]), int(rng[1]), int(rng[2])
    if seg_n != seg_end - seg_start + 1:
        raise SystemExit(f"Stage A segment is not contiguous ({seg_n} rows over {seg_start}..{seg_end})")
    print(f"    tracked segment: frames {seg_start}..{seg_end}  ({seg_n} of {n_frames})")

    # ---------------------------------------------------------------- 3. slice + Stage B
    n_cshard = args.contact_shards or len(gpus_contact)
    cranges = split_ranges(seg_n, n_cshard, minimum=32)
    if not done('stageB'):
        print(f"[3] slicing the segment into {len(cranges)} contact shards...")
        run_sync('slice_meta', [haco_py, STAGES / 'slice_meta.py', '--meta', stage_a,
                                '--outdir', work, '--prefix', tag, '--num_shards', len(cranges),
                                '--fps', fps, '--start-frame', seg_start, '--end-frame', seg_end],
                 logs / 'slice_meta.log', base_env(cfg))

    metas = sorted(work.glob(f'{tag}_shard*_meta.npz'))
    if not metas:
        raise SystemExit(f"no shard metas produced in {work}")
    contacts = [Path(str(m).replace('_meta.npz', '_contact.npz')) for m in metas]

    if done('stageB') and all(c.is_file() for c in contacts):
        print(f"[4] Stage B contact (cached, {len(contacts)} shards)")
    else:
        print(f"[4] Stage B: {args.model} contact inference, {len(metas)} shards over "
              f"GPUs {','.join(gpus_contact)}...")
        jobs = []
        for i, (m, c) in enumerate(zip(metas, contacts)):
            gpu = gpus_contact[i % len(gpus_contact)]
            cmd = [haco_py, STAGES / spec['script'], '--video', prepared, '--meta', m, '--out', c,
                   '--ckpt', ckpt, '--model_id', spec['model_id'], '--clip_len', clip_len,
                   '--stride', stride, '--batch', spec['batch'], '--mem_frac', mem_frac, '--thr', thr]
            if spec['img_size']:
                cmd += ['--img_size', spec['img_size']]
            jobs.append(Job(f"stageB.{i}", cmd, logs / f'stageB_{i}.log', base_env(cfg, gpu)))
        wait_all(jobs)
        mark('stageB')

    # ---------------------------------------------------------------- 5. wait for the tracker
    if track_job is not None:
        print('[5] waiting for HaWoR...')
        track_job.wait()
        mark('hawor')

    # ---------------------------------------------------------------- 6. filter
    track = work / ('track_filtered.npz' if args.filter == 'butterworth' else 'track_raw.npz')
    if done('filter') and track.is_file():
        print('[6] track filter (cached)')
    elif args.filter == 'butterworth':
        print(f"[6] filtering the hand track (zero-phase Butterworth, {args.cutoff_hz} Hz)...")
        run_sync('filter', [haco_py, STAGES / 'smooth_hand_tracking.py', '--mode', 'butterworth',
                            '--npz', hawor_raw, '--output', track,
                            '--report', work / 'filter_report.json',
                            '--cutoff-hz', args.cutoff_hz, '--order', 2],
                 logs / 'filter.log', base_env(cfg))
        mark('filter')
    else:
        print('[6] filter disabled (--filter none): using the raw track')
        track.write_bytes(hawor_raw.read_bytes())
        mark('filter')

    # ---------------------------------------------------------------- 7. combine
    combined = out / f'{tag}_combined.npz'
    if done('combine') and combined.is_file():
        print('[7] combine (cached)')
    else:
        print('[7] combining contact + track into one npz...')
        track_seg = work / 'track_segment.npz'
        run_sync('slice_track', [haco_py, STAGES / 'slice_hawor.py', '--npz', track, '--out', track_seg,
                                 '--start-frame', seg_start, '--end-frame', seg_end],
                 logs / 'slice_track.log', base_env(cfg))
        run_sync('combine', [haco_py, STAGES / 'haco_source.py',
                             '--contact-shards', *contacts, '--haco-meta', *metas,
                             '--hawor', track_seg,
                             '--haco-stage-b', STAGES / spec['script'], '--haco-model-id', spec['model_id'],
                             '--start-frame', seg_start, '--end-frame', seg_end, '--output', combined],
                 logs / 'combine.log', base_env(cfg))
        mark('combine')
    print(f"    combined npz -> {combined}")

    if args.skip_render:
        print(f"\ndone (render skipped) in {time.time() - t_start:.0f}s")
        return 0

    # ---------------------------------------------------------------- 8. render (sharded) + concat
    out_h264 = out / f'{tag}_meshtop.mp4'
    if done('render') and out_h264.is_file():
        print('[8] render (cached)')
    else:
        n_rshard = args.render_shards or len(gpus)
        rranges = split_ranges(seg_n, n_rshard, minimum=48)
        print(f"[8] rendering the mesh-top video: {len(rranges)} shards over GPUs {','.join(gpus)}...")
        shard_mp4s, jobs = [], []
        for i, (i0, i1) in enumerate(rranges):
            shard = work / f'render_shard{i:02d}.mp4'
            shard_mp4s.append(shard)
            cmd = [haco_py, STAGES / 'render_meshtop.py', '--combined', combined, '--video', prepared,
                   '--out_h264', shard,
                   '--sample_dir', out / 'samples', '--thr', thr, '--mesh_alpha', mesh_alpha,
                   '--start_idx', i0, '--end_idx', i1, '--hand', hand]
            jobs.append(Job(f"render.{i}", cmd, logs / f'render_{i}.log',
                            base_env(cfg, gpus[i % len(gpus)])))
        wait_all(jobs)

        if len(shard_mp4s) == 1:
            shard_mp4s[0].replace(out_h264)
        else:
            # The concat demuxer resolves `file '...'` RELATIVE to the list file's own directory,
            # so the list must hold ABSOLUTE paths. Every shard was written by the same encoder
            # settings at the same size and rate, so a stream copy is exact and instant.
            lst = work / 'concat.txt'
            lst.write_text(''.join(f"file '{p.resolve()}'\n" for p in shard_mp4s))
            run_sync('concat', ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0',
                                '-i', lst, '-c', 'copy', out_h264],
                     logs / 'concat.log', base_env(cfg))
        mark('render')

    # ---------------------------------------------------------------- summary
    summary = {
        'video': str(src_video), 'prepared': str(prepared), 'model': args.model,
        'model_label': spec['label'], 'checkpoint': str(ckpt), 'hand': hand,
        'fps': fps, 'frames_total': n_frames, 'segment': [seg_start, seg_end], 'segment_frames': seg_n,
        'thr': thr, 'mesh_alpha': mesh_alpha, 'filter': args.filter,
        'cutoff_hz': args.cutoff_hz if args.filter == 'butterworth' else None,
        'combined_npz': str(combined), 'video_out': str(out_h264),
        'wall_seconds': round(time.time() - t_start, 1),
    }
    (out / 'run_summary.json').write_text(json.dumps(summary, indent=2))
    print(f"\n=== done in {summary['wall_seconds']:.0f}s ===")
    print(f"  video    {out_h264}")
    print(f"  npz      {combined}")
    print(f"  samples  {out / 'samples'}")
    print(f"  summary  {out / 'run_summary.json'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
