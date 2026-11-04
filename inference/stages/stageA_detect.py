"""
STAGE A — find the hand in every frame and smooth its crop box.

  - read the video with cv2 (or a directory of numbered JPGs, see lib/utils/imgseq_capture.py),
  - run the WiLoR hand detector (your ONNX export of it, run with onnxruntime: lib/utils/hand_detector.py)
    on every frame and keep the most confident box of the requested side (class 1 = right hand,
    0 = left hand),
  - grow it to twice the hand's size about its centre (lib/utils/crop.py),
  - EMA-smooth the box over time (alpha=0.8, i.e. 80% of the previous box is kept each frame),
  - for every frame, also store a white-background 256x256 crop for quick inspection.

A frame where the detector finds no hand of that side reuses the last detected box; frames before the
first detection reuse the first one. Every frame of the video is kept, so the segment is the whole video.

Exports a single .npz that slice_meta.py shards for Stage B and the combine stage:
    frame_idx   : [N] int   absolute frame index in the video (0-based, cv2 order)
    bbox_sm     : [N,4]      EMA-smoothed [x,y,w,h], the box Stage B crops with
    detected    : [N] bool   whether the detector found the hand in that frame
    det_frame, det_xyxy, det_conf, det_cls : every raw detection, either hand (frame, x1 y1 x2 y2,
                  confidence, 1 = right / 0 = left); the HaWoR runner tracks hands from these
    crop_white  : [N,256,256,3] uint8 RGB  white-background crop
    fps, out_shape, img_wh
"""
import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm

from lib.utils.crop import expand_box, crop_transform, warp
from lib.utils.hand_detector import HandDetector, LEFT, RIGHT
from lib.utils.imgseq_capture import open_capture  # image-seq dir -> imread-faithful capture

DET_CONF = 0.3     # detector confidence threshold
DET_IOU = 0.3      # detector NMS IoU
BOX_ALPHA = 0.8    # EMA weight of the previous box
CROP_SIZE = 256    # size of the stored inspection crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--detector_path', default='data/base_data/demo_data/wilor_detector.onnx')
    ap.add_argument('--hand', default='right', choices=['right', 'left'], help='which hand to track/crop')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    detector = HandDetector(args.detector_path)
    print(f"[A] detector {args.detector_path} on {detector.providers[0]}", flush=True)
    side_cls = RIGHT if args.hand == 'right' else LEFT

    cap = open_capture(args.video)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not fps > 0:                                   # 0 or NaN from some containers
        fps = 30.0
    img_wh = np.asarray([int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))], np.int64)   # video WxH -> downstream projection
    n_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    print(f"[A] video={args.video} fps={fps:.4f}", flush=True)

    # pass 1: one detection per frame (the frames are kept for the crops below)
    frames, boxes = [], []
    dets = []                                          # every detection: (frame, x1, y1, x2, y2, conf, cls)
    with tqdm(total=n_hint, desc='[A] detect') as bar:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            found = detector(rgb, conf=DET_CONF, iou=DET_IOU)     # most confident first
            dets += [(len(frames), *d) for d in found]
            xyxy = next((d[:4] for d in found if d[5] == side_cls), None)
            frames.append(rgb)
            boxes.append(None if xyxy is None else expand_box(xyxy))
            bar.update(1)
    cap.release()
    n = len(frames)
    if n == 0:
        raise SystemExit(f"[A] could not read any frame from {args.video}")
    detected = np.array([b is not None for b in boxes])
    print(f"[A] {args.hand} hand detected in {int(detected.sum())}/{n} frames", flush=True)

    # fill the gaps: hold the last box; before the first detection use the first one; no detection at
    # all -> the whole frame (the contact map will then be meaningless, and the log says so)
    if detected.any():
        last = boxes[int(np.argmax(detected))]
    else:
        print("[A] WARNING: no hand detected in any frame — cropping the whole frame", flush=True)
        h, w = frames[0].shape[:2]
        last = expand_box((0, 0, w, h))
    for i in range(n):
        if boxes[i] is None:
            boxes[i] = last
        last = boxes[i]

    # pass 2: EMA over the boxes, then the inspection crops
    bbox_sm = np.zeros((n, 4), np.float32)
    crop_white = np.zeros((n, CROP_SIZE, CROP_SIZE, 3), np.uint8)
    state = None
    for i in tqdm(range(n), desc='[A] smooth+crop'):
        box = np.asarray(boxes[i], np.float64)
        state = box if state is None else BOX_ALPHA * state + (1.0 - BOX_ALPHA) * box
        bbox_sm[i] = state
        M = crop_transform(state, CROP_SIZE, CROP_SIZE)
        crop_white[i] = warp(frames[i], M, CROP_SIZE, CROP_SIZE, fill=255)

    frame_idx = np.arange(n, dtype=np.int64)
    det = np.asarray(dets, np.float32).reshape(-1, 7)
    np.savez_compressed(args.out, frame_idx=frame_idx, bbox_sm=bbox_sm, detected=detected,
                        det_frame=det[:, 0].astype(np.int64), det_xyxy=det[:, 1:5], det_conf=det[:, 5],
                        det_cls=det[:, 6],
                        crop_white=crop_white, fps=np.float32(fps),
                        out_shape=np.asarray((CROP_SIZE, CROP_SIZE), dtype=np.int64), img_wh=img_wh)
    print(f"[A] wrote {args.out}: N={n} segment=[0..{n - 1}] crop_white={crop_white.shape}", flush=True)


if __name__ == '__main__':
    main()
