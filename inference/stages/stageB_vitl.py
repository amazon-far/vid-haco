"""
STAGE B — ViT-L (V-JEPA 2.1 large/16-384) contact inference.

Identical clip construction / crop / normalization to the ViT-H Stage B (stageB_vith.py), EXCEPT:
  * imports the ViT-L model  lib.models.model_vjepa_vitl_seq.HACO_VJEPA  (embed 1024, DPT feats 384),
  * crops at 384x384 (the model's input res) instead of 256,
  * clip_len defaults to 8  (trained frames=8 stride=2),
  * --ckpt is OPTIONAL: with a checkpoint it loads state_dict STRICT (real epoch weights);
    with NO checkpoint it builds the hub backbone + RANDOM heads (for a code-path smoke).

Output .npz schema is byte-for-byte the same keys the render stage consumes:
    frame_idx [N] int, contact_prob [N,778], hand_contact [N,2], stride, clip_len, thr
"""
import os
import argparse
import numpy as np
import cv2
import torch
from tqdm import tqdm

from lib.models.model_vjepa_vitl_seq import HACO_VJEPA   # <-- ViT-L model (embed 1024, DPT feats 384)
from lib.utils.crop import crop_transform, warp
from lib.utils.imgseq_capture import open_capture   # image-seq dir -> imread-faithful capture

# ImageNet mean/std on [0,1] RGB, as in training; shaped to broadcast over CHW
VJEPA_MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
VJEPA_STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]


def clip_frame_indices(t, n_frames, clip_len, stride):
    idxs = [t - i * stride for i in range(clip_len)][::-1]   # oldest..t
    hi = max(0, n_frames - 1)
    return [min(max(j, 0), hi) for j in idxs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--meta', required=True)
    ap.add_argument('--ckpt', default='', help='epoch checkpoint; empty => random-init smoke model')
    ap.add_argument('--out', required=True)
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--clip_len', type=int, default=8)          # trained with frames=8
    ap.add_argument('--model_id', type=str, default='vjepa2.1-vitl-384')
    ap.add_argument('--img_size', type=int, default=384)        # ViT-L 2.1 input res (grid 24)
    ap.add_argument('--batch', type=int, default=4)             # smaller batch: 384^2 + shared GPU
    ap.add_argument('--thr', type=float, default=0.52)
    ap.add_argument('--mem_frac', type=float, default=0.35, help='cap this proc to frac of GPU mem (share w/ training)')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and 0.0 < args.mem_frac < 1.0:
        try:
            torch.cuda.set_per_process_memory_fraction(args.mem_frac, 0)  # bound footprint on the shared card
        except Exception as e:
            print(f"[B] set_per_process_memory_fraction skipped: {e}", flush=True)
    out_h = out_w = int(args.img_size)                           # 384x384 crops (matches training)

    meta = np.load(args.meta)
    seg_idx = meta['frame_idx']
    bbox_sm = meta['bbox_sm']
    N = len(seg_idx)
    print(f"[B-vitl] segment frames={N} stride={args.stride} clip_len={args.clip_len} img={out_h}", flush=True)

    # ---- decode every MOV frame we will ever need ----
    needed = set()
    for k in range(N):
        t = int(seg_idx[k])
        for j in clip_frame_indices(t, 10**9, args.clip_len, args.stride):
            needed.add(j)
    max_needed = max(needed)
    cap = open_capture(args.video)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = {}
    fi = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if fi in needed:
            frames[fi] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        fi += 1
        if fi > max_needed:
            break
    cap.release()
    print(f"[B-vitl] decoded {len(frames)} frames (video has {n_total})", flush=True)

    # ---- model (ViT-L) ----
    if args.ckpt and not os.path.isfile(args.ckpt):
        raise SystemExit(f"[B-vitl] checkpoint not found: {args.ckpt}")
    has_ckpt = bool(args.ckpt)
    print(f"[B-vitl] building HACO_VJEPA ViT-L (load_backbone_weights={not has_ckpt}, ckpt={'yes' if has_ckpt else 'SMOKE/random'})...", flush=True)
    model = HACO_VJEPA(freeze_backbone=True, binary_out=2, model_id=args.model_id,
                       img_size=args.img_size, load_backbone_weights=not has_ckpt).to(device)
    if has_ckpt:
        ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)  # CPU load: keep 13GB optimizer bulk off the GPU
        missing, unexpected = model.load_state_dict(ck['state_dict'], strict=True)
        assert not missing and not unexpected, (missing, unexpected)
        print(f"[B-vitl] ckpt STRICT (epoch={ck.get('epoch')}, "
              f"val c_f1={ck.get('val_metrics', {}).get('c_f1')})", flush=True)
    else:
        print("[B-vitl] SMOKE: random heads (contact predictions are meaningless)", flush=True)
    model.eval()

    def build_clip(k):
        """[clip_len,3,H,W] clip ending at segment frame k, every frame cropped with frame k's box."""
        t = int(seg_idx[k])
        M = crop_transform(bbox_sm[k], out_w, out_h)
        idxs = clip_frame_indices(t, n_total, args.clip_len, args.stride)
        clip = np.empty((args.clip_len, 3, out_h, out_w), dtype=np.float32)
        for c, fidx in enumerate(idxs):
            chw = warp(frames[fidx], M, out_w, out_h).transpose(2, 0, 1) / np.float32(255.0)
            clip[c] = (chw - VJEPA_MEAN) / VJEPA_STD
        return clip

    contact_prob = np.zeros((N, 778), dtype=np.float32)
    hand_contact = np.zeros((N, 2), dtype=np.float32)
    # 6-dim fingertip/palm force, present only for force-trained checkpoints (n_force>0). The head
    # regresses log1p(newtons), so we decode with expm1 here and store real NEWTONS.
    has_force = getattr(model, 'force_head', None) is not None
    force = np.zeros((N, getattr(model, 'n_force', 6)), dtype=np.float32) if has_force else None

    with torch.no_grad():
        for s in tqdm(range(0, N, args.batch), desc='[B-vitl] infer'):
            e = min(N, s + args.batch)
            batch = np.stack([build_clip(k) for k in range(s, e)])   # [b,8,3,384,384]
            clip_t = torch.from_numpy(batch).float().to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model({'input': {'image': clip_t}}, mode='test')
            contact_prob[s:e] = out['contact_out'].float().sigmoid().cpu().numpy()
            hand_contact[s:e] = out['hand_contact_out'].float().sigmoid().cpu().numpy()
            if has_force and 'force_out' in out:
                from lib.core.force_utils import decode_force
                force[s:e] = decode_force(
                    out['force_out'].float(),
                    out['force_logit'].float() if 'force_logit' in out else None).cpu().numpy()

    pred = contact_prob >= args.thr
    pf = pred.sum(1)
    print(f"[B-vitl] SANITY per-frame #contact (thr={args.thr}): min={pf.min()} mean={pf.mean():.1f} "
          f"max={pf.max()} | frames>=1: {(pf>0).sum()}/{N} | prob mean={contact_prob.mean():.3f}", flush=True)

    extra = {}
    if has_force:
        from lib.core.force_utils import ENCODING_VERSION
        extra['force'] = force        # [N, 6] newtons: thumb,index,middle,ring,pinky,palm
        # Stamp how these newtons were produced. The force encoding has already changed twice; a
        # consumer decoding an old file with a new decoder would silently rescale rather than fail.
        extra['force_encoding'] = np.asarray(ENCODING_VERSION)
        print(f"[B-vitl] FORCE head present: per-frame mean={force.mean():.4f} N max={force.max():.4f} N "
              f"| fingertip(0-4) mean={force[:, :5].mean():.4f} N", flush=True)
    np.savez_compressed(args.out, frame_idx=seg_idx, contact_prob=contact_prob,
                        hand_contact=hand_contact, stride=np.int64(args.stride),
                        clip_len=np.int64(args.clip_len), thr=np.float32(args.thr), **extra)
    print(f"[B-vitl] wrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
