#!/usr/bin/env python3
"""
Train (or evaluate) a Vid-HACO contact model on 8-frame clips.

    --model vith   V-JEPA 2 ViT-H/256 + DPT, 9 datasets          (released vith.pt = epoch 1 of this)
    --model vitl   V-JEPA 2.1 ViT-L/384 + DPT, 12 datasets       (released vitl.pt = epoch 6 of this)

Every hyperparameter defaults to the recipe in configs.py (read back from the original runs' own
checkpoints); any flag given on the command line overrides it. Run from a directory that holds data/
(the layout data_processing/README.md describes) with the repository root on PYTHONPATH:

  single GPU smoke   python training/train.py --model vitl --smoke 4 --batch 2 --workers 2
  8 GPUs             torchrun --nproc_per_node=8 training/train.py --model vith
  32 GPUs (4 nodes)  torchrun --nnodes=4 --nproc_per_node=8 --rdzv_backend=c10d \
                         --rdzv_endpoint=$MASTER_ADDR:29500 training/train.py --model vitl
  evaluate           python training/train.py --model vitl --eval_only --ckpt checkpoints/vitl.pt

Each epoch: train, validate on the held-out split (overall and per-dataset contact P/R/F1 at
--thres on the 778-vertex map, plus the binary head's accuracy / F1), save
<exp_dir>/checkpoints/haco_vjepa_alldata_epoch{E}.ckpt, update haco_vjepa_alldata_best.ckpt and
results.json. Re-running with the same --exp_dir resumes after the last saved epoch.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from configs import COMMON, RECIPES                       # noqa: E402
from losses import CONTACT_DATA, CONTACT_MEANS, ContactLoss  # noqa: E402
from models import build_model, split_params              # noqa: E402


# ----------------------------------------------------------------------------------------- arguments
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, choices=sorted(RECIPES))
    g = ap.add_argument_group('recipe (default: configs.py for --model)')
    g.add_argument('--model_id', type=str)
    g.add_argument('--input_size', type=int, help='square crop size fed to the backbone')
    g.add_argument('--datasets', type=str, help='comma list, in order')
    g.add_argument('--frames', type=int, help='clip length T (even: tubelet 2)')
    g.add_argument('--stride', type=int, help='frame stride inside a clip')
    g.add_argument('--batch', type=int, help='clips per GPU per micro-step')
    g.add_argument('--epochs', type=int)
    g.add_argument('--lr', type=float, help='decoder + heads')
    g.add_argument('--backbone_lr', type=float)
    g.add_argument('--lr_schedule', choices=['cosine', 'warmconst', 'constant'])
    o = ap.add_argument_group('training')
    o.add_argument('--exp_dir', type=str, help='default: experiments/<model>')
    o.add_argument('--freeze_backbone', type=int, default=0, help='1 = train decoder + heads only')
    o.add_argument('--warmup_steps', type=int, default=500, help='capped at total_steps // 20')
    o.add_argument('--grad_accum', type=int, default=1,
                   help='micro-steps per optimizer step; e.g. 4 on 8 GPUs reproduces vitl\'s 32-GPU step')
    o.add_argument('--weight_decay', type=float, default=COMMON['weight_decay'])
    o.add_argument('--amp', choices=['bf16', 'fp16', 'off'], default=COMMON['amp'])
    o.add_argument('--seed', type=int, default=COMMON['seed'])
    o.add_argument('--workers', type=int, default=10, help='dataloader workers per process')
    o.add_argument('--resized', type=int, default=1, help='read the half-resolution HOI4D frame tree')
    o.add_argument('--resize_factor', type=float, default=COMMON['resize_factor'])
    o.add_argument('--init_ckpt', type=str, default='', help='warm start: weights only, strict=False')
    o.add_argument('--vjepa2_ckpt', type=str, default='', help='vitl: V-JEPA 2.1 ViT-L/384 weights')
    o.add_argument('--log_every', type=int, default=200)
    o.add_argument('--smoke', type=int, default=0, help='run this many train steps + a tiny val, then exit')
    v = ap.add_argument_group('validation')
    v.add_argument('--thres', type=float, default=COMMON['thres'])
    v.add_argument('--val_batch', type=int, default=8)
    v.add_argument('--val_max', type=int, default=0, help='cap on val samples per rank (0 = all)')
    v.add_argument('--eval_only', action='store_true', help='validate --ckpt and exit')
    v.add_argument('--ckpt', type=str, default='', help='checkpoint for --eval_only')
    f = ap.add_argument_group('files')
    f.add_argument('--contact_data', type=str, default=CONTACT_DATA)
    f.add_argument('--contact_means', type=str, default=CONTACT_MEANS)
    f.add_argument('--mano_assets', type=str, default='',
                   help='default: inference/data/base_data/mano_assets.npz')
    a = ap.parse_args(argv)

    recipe = RECIPES[a.model]
    for k in ('model_id', 'input_size', 'frames', 'stride', 'batch', 'epochs', 'lr', 'backbone_lr',
              'lr_schedule'):
        if getattr(a, k) is None:
            setattr(a, k, recipe[k])
    a.datasets = [d for d in (a.datasets.split(',') if a.datasets else recipe['datasets']) if d]
    a.exp_dir = a.exp_dir or os.path.join('experiments', a.model)
    assert a.frames % 2 == 0, f"--frames must be even (tubelet 2), got {a.frames}"
    assert a.grad_accum >= 1
    if a.eval_only and not a.ckpt:
        ap.error('--eval_only needs --ckpt')
    return a


# ------------------------------------------------------------------------------------- distributed
RANK = int(os.environ.get('RANK', '0'))
LOCAL_RANK = int(os.environ.get('LOCAL_RANK', '0'))
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', '1'))
DIST = WORLD_SIZE > 1
IS_MAIN = RANK == 0
_LOG_FILE = None


def log(msg):
    if IS_MAIN:
        line = f"{time.strftime('%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        if _LOG_FILE is not None:
            _LOG_FILE.write(line + '\n')
            _LOG_FILE.flush()


def barrier():
    if DIST:
        dist.barrier()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lr_lambda_factory(schedule, total_steps, warmup_steps):
    warmup = min(warmup_steps, max(1, total_steps // 20))

    def f(step):
        if schedule == 'constant':
            return 1.0
        if step < warmup:
            return (step + 1) / warmup
        if schedule == 'warmconst':
            return 1.0
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    return f, warmup


def atomic_save(obj, path):
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


# -------------------------------------------------------------------------------------- validation
@torch.no_grad()
def validate(model, loader, device, ds_names, thres, amp_dtype, max_samples=0):
    """Overall + per-dataset contact P/R/F1 at `thres` on the 778-vertex map, and binary-head metrics.

    c_pre / c_rec / c_f1 are per-sample means (a sample counts toward precision if it predicts any
    contact, toward recall if it has any GT contact, toward F1 if both and P+R > 0); *_micro pool the
    vertex counts over the whole split. Summed across ranks before the ratios are taken."""
    model.eval()
    z = lambda *s: torch.zeros(*s, device=device, dtype=torch.float64)   # noqa: E731
    nd = len(ds_names)
    acc = dict(tp=z(()), pp=z(()), gp=z(()), s_pre=z(()), s_rec=z(()), s_f1=z(()), n_pre=z(()),
               n_rec=z(()), n_f1=z(()), n=z(()), b_ok=z(()), b_n=z(()), b_tp=z(()), b_pp=z(()), b_gp=z(()),
               d_tp=z(nd), d_pp=z(nd), d_gp=z(nd), d_n=z(nd), d_sf1=z(nd), d_nf1=z(nd), d_bok=z(nd), d_bn=z(nd))
    seen = 0
    for data in loader:
        clip = data['input_data']['image']
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model({'input': {'image': clip}}, mode='test')
        pred = (out['contact_out'].float().sigmoid() >= thres).float()                 # [B, 778]
        gt = data['targets_data']['contact_data']['contact_h'].to(device).float()
        ds = data['meta_info']['dataset_id'].to(device).long()

        tp_s, pp_s, gp_s = (pred * gt).sum(1), pred.sum(1), gt.sum(1)
        pre_ok, rec_ok = pp_s > 0, gp_s > 0
        pre = torch.where(pre_ok, tp_s / pp_s.clamp(min=1), torch.zeros_like(tp_s))
        rec = torch.where(rec_ok, tp_s / gp_s.clamp(min=1), torch.zeros_like(tp_s))
        f1_ok = pre_ok & rec_ok & ((pre + rec) > 0)
        f1 = torch.where(f1_ok, 2 * pre * rec / (pre + rec).clamp(min=1e-9), torch.zeros_like(tp_s))

        acc['tp'] += tp_s.sum(); acc['pp'] += pp_s.sum(); acc['gp'] += gp_s.sum()
        acc['s_pre'] += pre[pre_ok].sum(); acc['n_pre'] += pre_ok.sum()
        acc['s_rec'] += rec[rec_ok].sum(); acc['n_rec'] += rec_ok.sum()
        acc['s_f1'] += f1[f1_ok].sum(); acc['n_f1'] += f1_ok.sum()
        acc['n'] += gt.shape[0]
        acc['d_tp'].index_add_(0, ds, tp_s.double()); acc['d_pp'].index_add_(0, ds, pp_s.double())
        acc['d_gp'].index_add_(0, ds, gp_s.double()); acc['d_n'].index_add_(0, ds, torch.ones_like(tp_s).double())
        acc['d_sf1'].index_add_(0, ds, f1.double()); acc['d_nf1'].index_add_(0, ds, f1_ok.double())

        hp = (out['hand_contact_out'].float().sigmoid() > 0.5).float()
        hg = data['targets_data']['hand_contact_gt'].to(device).float()
        acc['b_ok'] += (hp == hg).sum(); acc['b_n'] += hg.numel()
        acc['b_tp'] += (hp * hg).sum(); acc['b_pp'] += hp.sum(); acc['b_gp'] += hg.sum()
        acc['d_bok'].index_add_(0, ds, (hp == hg).sum(1).double())
        acc['d_bn'].index_add_(0, ds, torch.full_like(tp_s, hg.shape[1]).double())

        seen += clip.shape[0]
        if max_samples and seen >= max_samples:
            break

    if DIST:
        for t in acc.values():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    a = {k: (v.item() if v.dim() == 0 else v.tolist()) for k, v in acc.items()}
    div = lambda x, y: x / y if y > 0 else 0.0                                       # noqa: E731
    f1_of = lambda p, r: 2 * p * r / (p + r) if (p + r) > 0 else 0.0                 # noqa: E731
    mp, mr = div(a['tp'], a['pp']), div(a['tp'], a['gp'])
    bp, br = div(a['b_tp'], a['b_pp']), div(a['b_tp'], a['b_gp'])
    per = {}
    for i, name in enumerate(ds_names):
        dp, dr = div(a['d_tp'][i], a['d_pp'][i]), div(a['d_tp'][i], a['d_gp'][i])
        per[name] = dict(n=int(a['d_n'][i]), c_f1=div(a['d_sf1'][i], a['d_nf1'][i]), c_f1_micro=f1_of(dp, dr),
                         bin_acc=div(a['d_bok'][i], a['d_bn'][i]))
    return dict(c_pre=div(a['s_pre'], a['n_pre']), c_rec=div(a['s_rec'], a['n_rec']), c_f1=div(a['s_f1'], a['n_f1']),
                c_pre_micro=mp, c_rec_micro=mr, c_f1_micro=f1_of(mp, mr),
                bin_acc=div(a['b_ok'], a['b_n']), bin_f1=f1_of(bp, br), n_val=int(a['n']), per_dataset=per)


def log_val(vm, prefix='VAL'):
    log(f"  {prefix} (n={vm['n_val']}) C-Pre={vm['c_pre']:.4f} C-Rec={vm['c_rec']:.4f} C-F1={vm['c_f1']:.4f} "
        f"| micro C-F1={vm['c_f1_micro']:.4f} | bin_acc={vm['bin_acc']:.4f} bin_f1={vm['bin_f1']:.4f}")
    for n, d in vm['per_dataset'].items():
        log(f"    {n:14s} n={d['n']:7d} C-F1={d['c_f1']:.4f} (micro {d['c_f1_micro']:.4f}) bin_acc={d['bin_acc']:.4f}")


# -------------------------------------------------------------------------------------------- main
def main(argv=None):
    global _LOG_FILE
    args = parse_args(argv)

    if DIST:
        # generous timeout: the other ranks wait in barrier() while rank 0 builds split caches on a
        # first run or writes multi-GB checkpoints to shared storage
        # VIDHACO_DIST_BACKEND=gloo lets several ranks share one GPU (for testing the DDP code paths)
        dist.init_process_group(backend=os.environ.get('VIDHACO_DIST_BACKEND', 'nccl'), init_method='env://',
                                timeout=timedelta(hours=2))
        gpu = LOCAL_RANK % torch.cuda.device_count()
        torch.cuda.set_device(gpu)
        device = torch.device(f'cuda:{gpu}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    ckpt_dir = os.path.join(args.exp_dir, 'checkpoints')
    if IS_MAIN:
        os.makedirs(ckpt_dir, exist_ok=True)
        _LOG_FILE = open(os.path.join(args.exp_dir, 'eval.log' if args.eval_only else 'train.log'), 'a')
    barrier()

    seed_everything(args.seed + RANK)
    torch.set_num_threads(max(1, args.workers))
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'off': None}[args.amp]

    log(f"[train] model={args.model} backbone={args.model_id} input={args.input_size} frames={args.frames} "
        f"stride={args.stride} world_size={WORLD_SIZE} batch/gpu={args.batch} grad_accum={args.grad_accum} "
        f"-> {args.batch * WORLD_SIZE * args.grad_accum} clips/step | epochs={args.epochs} lr={args.lr:g} "
        f"backbone_lr={args.backbone_lr:g} schedule={args.lr_schedule} amp={args.amp}")
    rec_ws = RECIPES[args.model]['world_size']
    if not args.eval_only and args.batch * WORLD_SIZE * args.grad_accum != RECIPES[args.model]['batch'] * rec_ws:
        log(f"[train] NOTE: the released run stepped on {RECIPES[args.model]['batch'] * rec_ws} clips "
            f"({RECIPES[args.model]['batch']}/GPU x {rec_ws} GPUs)")

    # ---- data. Missing split caches get written on first use, so build them one writer at a time:
    #      global rank 0 first, then local rank 0 of the other nodes (a no-op when data/ is shared). ----
    from data_processing.multidata_vjepa import MultiDataVJEPA
    ds_kw = dict(clip_len=args.frames, clip_stride=args.stride, use_resized=bool(args.resized),
                 resize_factor=args.resize_factor, datasets=args.datasets, input_size=args.input_size)
    for writer in (IS_MAIN, LOCAL_RANK == 0 and not IS_MAIN):
        if writer:
            if not args.eval_only:
                MultiDataVJEPA('train', verbose=IS_MAIN, **ds_kw)
            MultiDataVJEPA('val', verbose=IS_MAIN, **ds_kw)
        barrier()
    val_set = MultiDataVJEPA('val', verbose=False, **ds_kw)
    ds_names = val_set.names
    val_loader = DataLoader(val_set, batch_size=args.val_batch, shuffle=False,
                            sampler=DistributedSampler(val_set, shuffle=False) if DIST else None,
                            num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    log(f"[data] val={len(val_set)} {val_set.per_dataset_counts}")

    # ---- model ----
    loading_ckpt = bool(args.eval_only and args.ckpt)
    model = build_model(args.model, args.model_id, freeze_backbone=bool(args.freeze_backbone) or args.eval_only,
                        input_size=args.input_size, vjepa2_ckpt=args.vjepa2_ckpt or None,
                        load_backbone_weights=not loading_ckpt, log=log).to(device)

    if args.eval_only:
        ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        sd = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck
        model.load_state_dict(sd, strict=True)
        log(f"[eval] loaded {args.ckpt} (strict) epoch={ck.get('epoch', '?') if isinstance(ck, dict) else '?'}")
        vm = validate(model, val_loader, device, ds_names, args.thres, amp_dtype, args.val_max)
        log_val(vm)
        ref = ck.get('val_metrics') if isinstance(ck, dict) else None
        if IS_MAIN and ref:
            for n, d in vm['per_dataset'].items():
                r = ref.get('per_dataset', {}).get(n)
                if r and d['n']:
                    log(f"    {n:14s} C-F1 {d['c_f1']:.4f} vs checkpoint {r['c_f1']:.4f} "
                        f"(n {d['n']} vs {r['n']}) bin_acc {d['bin_acc']:.4f} vs {r['bin_acc']:.4f}")
            with open(os.path.join(args.exp_dir, 'eval.json'), 'w') as fh:
                json.dump(dict(ckpt=args.ckpt, metrics=vm, checkpoint_metrics=ref), fh, indent=2)
        if DIST:
            dist.destroy_process_group()
        return

    train_set = MultiDataVJEPA('train', verbose=False, **ds_kw)
    log(f"[data] train={len(train_set)} {train_set.per_dataset_counts}")
    train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=True) if DIST else None
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=train_sampler is None,
                              sampler=train_sampler, num_workers=args.workers, pin_memory=True, drop_last=True,
                              generator=gen, persistent_workers=args.workers > 0)

    core = model
    bb, head = split_params(core)
    groups = ([{'params': bb, 'lr': args.backbone_lr}] if bb else []) + [{'params': head, 'lr': args.lr}]
    optimizer = torch.optim.AdamW(groups, betas=COMMON['betas'], weight_decay=args.weight_decay)
    log(f"[model] trainable backbone {sum(p.numel() for p in bb) / 1e6:.1f}M ({len(bb)} tensors, lr {args.backbone_lr:g}) "
        f"| decoder+heads {sum(p.numel() for p in head) / 1e6:.1f}M ({len(head)} tensors, lr {args.lr:g})")

    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    lr_fn, warmup = lr_lambda_factory(args.lr_schedule, total_steps, args.warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp == 'fp16')
    log(f"[train] {steps_per_epoch} steps/epoch, {total_steps} total, warmup {warmup}")

    from lib.utils.mano_assets import DEFAULT_PATH as MANO_ASSETS, load as load_mano_assets
    faces = load_mano_assets(args.mano_assets or MANO_ASSETS)['faces']
    criterion = ContactLoss(faces, args.epochs, args.contact_data, args.contact_means).to(device)

    # ---- resume (latest epoch checkpoint in exp_dir) or warm start, BEFORE the DDP wrap. Every rank
    #      loads the same file, so exp_dir must be shared storage; rank 0's view is checked on all. ----
    results, best = [], {'c_f1': -1.0, 'epoch': -1, 'path': ''}
    start_epoch, global_step = 0, 0
    done = [e for e in range(args.epochs) if os.path.isfile(os.path.join(ckpt_dir, f'haco_vjepa_alldata_epoch{e}.ckpt'))]
    if DIST:
        ref = [done]
        dist.broadcast_object_list(ref, src=0)
        if ref[0] != done:
            raise RuntimeError(f"rank {RANK} sees epoch checkpoints {done} in {ckpt_dir}, rank 0 sees {ref[0]}: "
                               f"--exp_dir must be on storage shared by every node")
    if done:
        path = os.path.join(ckpt_dir, f'haco_vjepa_alldata_epoch{done[-1]}.ckpt')
        ck = torch.load(path, map_location='cpu', weights_only=False)
        core.load_state_dict(ck['state_dict'])
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        if ck.get('scaler'):          # empty when saved from a bf16 / fp32 run
            scaler.load_state_dict(ck['scaler'])
        start_epoch, global_step = ck['epoch'] + 1, ck.get('global_step', 0)
        results, best = ck.get('results'), ck.get('best')
        rj = os.path.join(args.exp_dir, 'results.json')
        if (results is None or best is None) and os.path.isfile(rj):   # checkpoints of the original trainer
            with open(rj) as fh:
                prior = json.load(fh)
            results = prior.get('per_epoch', []) if results is None else results
            best = prior.get('best') if best is None else best
        results = results or []
        best = best or {'c_f1': -1.0, 'epoch': -1, 'path': ''}
        log(f"[train] resumed from {path} -> epoch {start_epoch}, step {global_step}, best so far {best}")
        del ck
    elif args.init_ckpt:
        ck = torch.load(args.init_ckpt, map_location='cpu', weights_only=False)
        missing, unexpected = core.load_state_dict(ck.get('state_dict', ck), strict=False)
        log(f"[train] warm start from {args.init_ckpt}: missing={len(missing)} unexpected={len(unexpected)}")
        del ck

    if DIST:
        model = DDP(model, device_ids=[device.index], output_device=device.index,
                    **({'find_unused_parameters': True} if args.freeze_backbone else {'static_graph': True}))
    barrier()

    # ---- train ----
    # DDP(static_graph=True) records the graph on its first synchronized backward; a no_sync() backward
    # before that trips an internal assert. So the first micro-step of the run always all-reduces
    # (exact: those gradients are just averaged early, and averaging is linear).
    primed = not DIST
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        t0, t_data, last = time.time(), 0.0, None
        t_fetch = time.time()
        for it, data in enumerate(train_loader):
            t_data += time.time() - t_fetch
            micro = it % args.grad_accum
            sync = micro == args.grad_accum - 1
            if micro == 0:
                optimizer.zero_grad(set_to_none=True)
            with (model.no_sync() if DIST and primed and not sync else _null()):
                with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_dtype is not None):
                    out = model({'input': {'image': data['input_data']['image']}}, mode='train')
                loss, ld = criterion(out, data['targets_data'], epoch, data['meta_info'].get('has_seg'))
                scaler.scale(loss / args.grad_accum).backward()
            primed = True
            last = ld
            if not sync:
                t_fetch = time.time()
                continue
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            step = it // args.grad_accum

            if step % args.log_every == 0:
                lrs = '/'.join(f'{x:.2e}' for x in scheduler.get_last_lr())
                log(f"E{epoch} it{step}/{steps_per_epoch} step{global_step} lr(backbone/head)={lrs} "
                    f"total={ld['total_loss'].item():.4f} map={ld['contact_map_loss'].item():.4f} "
                    f"bin={ld['binary_loss'].item():.4f} seg={ld['seg_loss'].item():.4f} "
                    f"bin_acc={ld['binary_acc'].item():.3f} data_s/it={t_data / (it + 1):.3f}")
            if args.smoke:
                gn = lambda ps: sum(float(p.grad.float().norm() ** 2) for p in ps if p.grad is not None) ** 0.5  # noqa: E731
                ok = bool(torch.isfinite(loss)) and math.isfinite(gn(bb + head))
                log(f"[smoke] step{step} loss={loss.item():.4f} finite={ok} grad_norm(backbone)={gn(bb):.4f} "
                    f"grad_norm(head)={gn(head):.4f} clip={tuple(data['input_data']['image'].shape)} "
                    f"contact_out={tuple(out['contact_out'].shape)} datasets={data['meta_info']['dataset_id'].tolist()}")
                if step + 1 >= args.smoke:
                    mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                    log(f"[smoke] {step + 1} steps, peak GPU memory {mem:.1f} GB; tiny val pass")
                    log_val(validate(model, val_loader, device, ds_names, args.thres, amp_dtype,
                                     max_samples=max(2 * args.val_batch, 16)), prefix='SMOKE VAL')
                    if DIST:
                        dist.destroy_process_group()
                    return
            t_fetch = time.time()

        train_secs = time.time() - t0
        t_val = time.time()
        vm = validate(model, val_loader, device, ds_names, args.thres, amp_dtype, args.val_max)
        val_secs = time.time() - t_val
        log(f"=== EPOCH {epoch} === train {train_secs:.0f}s (data {t_data:.0f}s) val {val_secs:.0f}s "
            f"peak_mem {torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0:.1f}GB")
        log_val(vm)

        lrs = scheduler.get_last_lr()      # [backbone, head], or [head] with --freeze_backbone 1
        results.append(dict(epoch=epoch, global_step=global_step, lr=lrs[0], lr_head=lrs[-1],
                            train_secs=round(train_secs, 1), val_secs=round(val_secs, 1),
                            train_loss={k: float(v) for k, v in (last or {}).items()},
                            **{k: v for k, v in vm.items()}))
        path = os.path.join(ckpt_dir, f'haco_vjepa_alldata_epoch{epoch}.ckpt')
        if vm['c_f1'] > best['c_f1']:
            best = {'c_f1': vm['c_f1'], 'epoch': epoch, 'path': path}
        if IS_MAIN:
            atomic_save({'epoch': epoch, 'global_step': global_step, 'state_dict': core.state_dict(),
                         'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                         'scaler': scaler.state_dict(), 'val_metrics': vm, 'args': vars(args), 'datasets': ds_names,
                         'results': results, 'best': best}, path)
            log(f"[train] saved {path}")
            if best['epoch'] == epoch:
                atomic_save({'epoch': epoch, 'state_dict': core.state_dict(), 'val_metrics': vm},
                            os.path.join(ckpt_dir, 'haco_vjepa_alldata_best.ckpt'))
                log(f"[train] new best val C-F1 {vm['c_f1']:.4f} @ epoch {epoch}")
            with open(os.path.join(args.exp_dir, 'results.json'), 'w') as fh:
                json.dump(dict(args=vars(args), world_size=WORLD_SIZE, datasets=ds_names,
                               train_per_dataset=train_set.per_dataset_counts,
                               val_per_dataset=val_set.per_dataset_counts,
                               per_epoch=results, best=best), fh, indent=2)
        barrier()

    log(f"[train] done. best val C-F1 {best['c_f1']:.4f} @ epoch {best['epoch']}")
    if DIST:
        dist.destroy_process_group()


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


if __name__ == '__main__':
    main()
