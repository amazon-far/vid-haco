#!/usr/bin/env python3
"""
Check that everything this project needs is reachable, before a long run discovers it isn't.

Verifies the project files and the local assets you set up (MANO assets, WiLoR detector ONNX export, V-JEPA 2 clone),
the four external resources named in config.env (two environments, the tracker, the HF cache), the two
checkpoints, ffmpeg, and the GPUs
— and for the environments it actually imports the modules the stages import, rather than just
checking that a python binary exists.

    ./doctor.py            # check everything
    ./doctor.py --model vitl   # only what the ViT-L path needs
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import MODELS, ROOT, base_env, load_config   # noqa: E402

OK, BAD, WARN = '  ok  ', ' FAIL ', ' warn '
state = {'fail': 0, 'warn': 0}


def report(status, what, detail=''):
    if status is BAD:
        state['fail'] += 1
    elif status is WARN:
        state['warn'] += 1
    print(f"[{status}] {what}" + (f"  — {detail}" if detail else ''))


def check_path(label, path, kind='file'):
    p = Path(path) if path else None
    if not path:
        return report(BAD, label, 'not configured')
    exists = p.is_dir() if kind == 'dir' else p.is_file()
    if not exists:
        return report(BAD, label, f"missing: {p}")
    size = ''
    if kind == 'file':
        mb = p.stat().st_size / 1e6
        size = f"{mb / 1000:.2f} GB" if mb > 1000 else f"{mb:.1f} MB"
    report(OK, label, f"{p}{'  ' + size if size else ''}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='', choices=[''] + sorted(MODELS),
                    help='only check what this model needs (default: both)')
    ap.add_argument('--set', dest='overrides', action='append', default=[], metavar='KEY=VALUE')
    args = ap.parse_args()
    cfg = load_config(ROOT / 'config.env', args.overrides)
    models = [args.model] if args.model else sorted(MODELS)

    print('--- project files and local assets ---')
    for label, rel, kind in [
        ('contact model (ViT-L)', 'lib/models/model_vjepa_vitl_seq.py', 'file'),
        ('contact model (ViT-H)', 'lib/models/model_vjepa_seq.py', 'file'),
        ('MANO assets', 'data/base_data/mano_assets.npz', 'file'),
        ('WiLoR detector (your ONNX export)', 'data/base_data/demo_data/wilor_detector.onnx', 'file'),
        ('V-JEPA 2 repo', 'third_party/vjepa2_repo/src/hub/backbones.py', 'file'),
        ('Stage A', 'stages/stageA_detect.py', 'file'),
        ('renderer', 'stages/render_meshtop.py', 'file'),
        ('combiner', 'stages/haco_source.py', 'file'),
        ('smoother', 'stages/hand_smoother.py', 'file'),
    ]:
        check_path(label, ROOT / rel, kind)

    print('\n--- tools ---')
    for tool in ('ffmpeg', 'ffprobe'):
        path = shutil.which(tool)
        report(OK if path else BAD, tool, path or 'not on PATH')

    print('\n--- environments ---')
    haco_py = cfg.get('HACO_PY', '')
    if check_path('haco_render python', haco_py):
        probe = '\n'.join([
            "import json",
            "mods = {}",
            "for m in ['torch','cv2','numpy','scipy','trimesh','pyrender','transformers','onnxruntime','tqdm']:",
            "    try:",
            "        mods[m] = getattr(__import__(m), '__version__', 'present')",
            "    except Exception as e:",
            "        mods[m] = 'MISSING: %s' % e",
            "try:",
            "    import torch",
            "    mods['cuda'] = '%s (%d devices)' % (torch.version.cuda, torch.cuda.device_count())",
            "except Exception as e:",
            "    mods['cuda'] = 'MISSING: %s' % e",
            "print(json.dumps(mods))",
        ])
        try:
            out = subprocess.run([haco_py, '-c', probe], capture_output=True, text=True,
                                 env=base_env(cfg), cwd=ROOT, timeout=300)
            if out.returncode != 0:
                report(BAD, 'haco_render imports', out.stderr.strip().splitlines()[-1:] or 'failed')
            else:
                import json as _json
                mods = _json.loads(out.stdout.strip().splitlines()[-1])
                for m, v in mods.items():
                    report(BAD if str(v).startswith('MISSING') else OK, f"  {m}", str(v))
        except subprocess.TimeoutExpired:
            report(BAD, 'haco_render imports', 'timed out')

        # The stages import the project's lib with the project root as cwd; prove that works.
        lib_probe = ("from lib.utils import mano_assets;"
                     "from lib.utils.crop import crop_transform;"
                     "from lib.utils.contact_mask import drop_small_components;"
                     "a = mano_assets.load();"
                     "print('verts', a['v_template'].shape[0], 'faces', a['watertight_faces'].shape[0])")
        out = subprocess.run([haco_py, '-c', lib_probe], capture_output=True, text=True,
                             env=base_env(cfg), cwd=ROOT)
        report(OK if out.returncode == 0 and 'verts 778' in out.stdout else BAD,
               'project lib imports (crop, contact mask, MANO assets)',
               'ok' if out.returncode == 0 else (out.stderr.strip().splitlines() or [''])[-1])

    hawor_py, hawor_root = cfg.get('HAWOR_PY', ''), cfg.get('HAWOR_ROOT', '')
    check_path('hawor python', hawor_py)
    check_path('HaWoR checkout', hawor_root, 'dir')
    if hawor_root:
        check_path('  run_hawor_headless.py', Path(hawor_root) / 'run_hawor_headless.py')
        check_path('  hawor.ckpt', Path(hawor_root) / 'weights/hawor/checkpoints/hawor.ckpt')
        check_path('  infiller.pt', Path(hawor_root) / 'weights/hawor/checkpoints/infiller.pt')

    print('\n--- checkpoints ---')
    for m in models:
        spec = MODELS[m]
        ckpt = cfg.get(spec['ckpt_key'], '')
        if ckpt and not Path(ckpt).is_absolute():
            ckpt = str(ROOT / ckpt)   # config.env paths are relative to the project root, as in run.py
        check_path(f"--model {m}  ({spec['label']})", ckpt)

    if any(MODELS[m]['needs_hf_cache'] for m in models):
        print('\n--- HF cache (ViT-H backbone only) ---')
        hub = Path(cfg.get('HF_CACHE', ''))
        snap = hub / 'models--facebook--vjepa2-vith-fpc64-256'
        report(OK if snap.is_dir() else BAD, 'facebook/vjepa2-vith-fpc64-256',
               str(snap) if snap.is_dir() else f"not cached under {hub} — the ViT-H path needs it "
                                               f"(the ViT-L path does not)")

    print('\n--- GPUs ---')
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total',
                              '--format=csv,noheader'], capture_output=True, text=True, check=True)
        lines = [l for l in out.stdout.strip().splitlines() if l.strip()]
        want = [g.strip() for g in cfg.get('GPUS', '').split(',') if g.strip()]
        report(OK, f"{len(lines)} GPU(s) visible", f"configured GPUS={','.join(want)}")
        for l in lines:
            idx = l.split(',')[0].strip()
            print(f"        {'*' if idx in want else ' '} {l}")
        missing = [g for g in want if g not in {l.split(',')[0].strip() for l in lines}]
        if missing:
            report(BAD, 'configured GPUs not present', ','.join(missing))
        busy = [l for l in lines if int(l.split(',')[2].strip().split()[0]) > 2000]
        if busy:
            report(WARN, f"{len(busy)} GPU(s) already in use",
                   'lower MEM_FRAC (config.env) so the pipeline co-exists with them')
    except Exception as e:
        report(BAD, 'nvidia-smi', str(e))

    print()
    if state['fail']:
        print(f"{state['fail']} problem(s) found — fix these before running.")
        return 1
    print('all checks passed' + (f", {state['warn']} warning(s)" if state['warn'] else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
