"""
STAGE B — ViT-H (V-JEPA 2 huge/16-256, transformers VJEPA2Model) contact inference.

Reads the video + Stage-A per-frame smoothed bboxes, and for every target frame t in the segment
builds a V-JEPA clip ENDING at t, then runs the frozen-V-JEPA contact model to get per-frame
778-vertex contact logits + the 2-bit hand-contact logits.

CLIP CONSTRUCTION -- matches the training clip loader (data_processing/loaders/HOI4D/dataset_vjepa.py):
  * Frame indices ending at t: [t-(clip_len-1)*stride, ..., t-stride, t]; negative indices are
    clamped to 0 (repeat frame 0).
  * The crop affine is computed ONCE from the TARGET frame t's bbox (lib/utils/crop.py: box centre to
    crop centre, each side scaled to the crop) and applied IDENTICALLY (cv2.warpAffine, INTER_LINEAR,
    border 0) to every frame of the clip. No flip / colour augmentation at test time.
  * Each cropped 256x256 frame -> CHW/255 -> ImageNet mean/std. Clip tensor [1,clip_len,3,256,256].
  Difference vs training: the dataset reads pre-decoded JPGs; here clip frames are pulled from the
  video with cv2 at the matching absolute indices. The crop transform / normalization are identical.

run.py passes --clip_len 8 --stride 2, which is how the released checkpoint was trained.

GPU: V-JEPA inference is ~5 GB; clips are batched (default 8) under torch.no_grad().

Output .npz:
    frame_idx       : [N] int    same absolute video indices as Stage A (segment order)
    contact_prob    : [N,778]    sigmoid(contact_out)        (so the render stage can threshold)
    hand_contact    : [N,2]      sigmoid(hand_contact_out)   ([left,right])
    stride, clip_len, thr
"""
import os
import argparse
import numpy as np
import cv2
import torch
from tqdm import tqdm

from lib.models.model_vjepa_seq import HACO_VJEPA
from lib.utils.crop import crop_transform, warp

IMG_SIZE = 256     # the ViT-H model's input resolution
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
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--clip_len', type=int, default=8)
    ap.add_argument('--model_id', type=str, default='facebook/vjepa2-vith-fpc64-256',
                    help='V-JEPA 2 backbone id (Hugging Face); must match the checkpoint')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--thr', type=float, default=0.52)
    ap.add_argument('--mem_frac', type=float, default=0.9, help='cap this proc to frac of GPU mem (dense packing)')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and 0.0 < args.mem_frac < 1.0:
        try:
            torch.cuda.set_per_process_memory_fraction(args.mem_frac, 0)
        except Exception as e:
            print(f"[B] set_per_process_memory_fraction skipped: {e}", flush=True)
    out_h = out_w = IMG_SIZE

    meta = np.load(args.meta)
    seg_idx = meta['frame_idx']      # [N] absolute video frame indices, segment order
    bbox_sm = meta['bbox_sm']        # [N,4] smoothed bbox per target frame
    N = len(seg_idx)
    print(f"[B] segment frames={N} stride={args.stride} clip_len={args.clip_len}", flush=True)

    # ---- decode every video frame we will ever need (segment frames + their clip histories) ----
    needed = set()
    for k in range(N):
        t = int(seg_idx[k])
        for j in clip_frame_indices(t, 10**9, args.clip_len, args.stride):
            needed.add(j)
    max_needed = max(needed)
    print(f"[B] decoding {len(needed)} unique video frames (max idx {max_needed})...", flush=True)
    cap = cv2.VideoCapture(args.video)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = {}
    fi = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if fi in needed:
            frames[fi] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)  # RGB [0,255]
        fi += 1
        if fi > max_needed:
            break
    cap.release()
    print(f"[B] decoded {len(frames)} frames (video has {n_total})", flush=True)

    # ---- model ----
    print("[B] loading HACO_VJEPA (freeze_backbone=True, binary_out=2)...", flush=True)
    model = HACO_VJEPA(freeze_backbone=True, binary_out=2, model_id=args.model_id).to(device)
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)  # CPU load: keep optimizer bulk off GPU
    missing, unexpected = model.load_state_dict(ck['state_dict'], strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    print(f"[B] checkpoint loaded STRICT (epoch={ck.get('epoch')}, "
          f"val c_f1={ck.get('val_metrics', {}).get('c_f1')})", flush=True)
    model.eval()

    def build_clip(k):
        """[clip_len,3,256,256] clip ending at segment frame k, every frame cropped with frame k's box."""
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

    with torch.no_grad():
        for s in tqdm(range(0, N, args.batch), desc='[B] vjepa infer'):
            e = min(N, s + args.batch)
            batch = np.stack([build_clip(k) for k in range(s, e)])     # [b,clip_len,3,256,256]
            clip_t = torch.from_numpy(batch).float().to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model({'input': {'image': clip_t}}, mode='test')
            contact_prob[s:e] = out['contact_out'].float().sigmoid().cpu().numpy()
            hand_contact[s:e] = out['hand_contact_out'].float().sigmoid().cpu().numpy()

    # ---- sanity ----
    pred = contact_prob >= args.thr
    per_frame_pos = pred.sum(1)
    print(f"[B] SANITY  per-frame #contact-verts (thr={args.thr}): "
          f"min={per_frame_pos.min()} mean={per_frame_pos.mean():.1f} max={per_frame_pos.max()} "
          f"| frames with >=1 contact: {(per_frame_pos>0).sum()}/{N}", flush=True)
    print(f"[B] SANITY  contact_prob: min={contact_prob.min():.3f} "
          f"mean={contact_prob.mean():.3f} max={contact_prob.max():.3f}", flush=True)
    print(f"[B] SANITY  hand_contact[right] mean={hand_contact[:,1].mean():.3f} "
          f"left mean={hand_contact[:,0].mean():.3f}", flush=True)
    if per_frame_pos.max() == 0:
        print("[B] WARNING: all-zero contact predictions!", flush=True)
    if (per_frame_pos == 778).all():
        print("[B] WARNING: all-one contact predictions!", flush=True)

    np.savez_compressed(args.out, frame_idx=seg_idx, contact_prob=contact_prob,
                        hand_contact=hand_contact, stride=np.int64(args.stride),
                        clip_len=np.int64(args.clip_len), thr=np.float32(args.thr))
    print(f"[B] wrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
