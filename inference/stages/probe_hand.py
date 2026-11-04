#!/usr/bin/env python3
"""
Decide which hand a video is about, so `--hand auto` can pick without the user looking.

This matters more than it looks. The contact model is right-canonical (training mirrors every
left hand so the crop reads as a right hand) and HaWoR tracks right hands only, so a left-handed
video must be mirrored — see prepare_video.py. Worse, choosing wrong does NOT fail loudly:
Stage A looks for the requested side only, and when it never finds one it crops the whole frame,
so the run completes with the model looking at the entire frame instead of a hand, and the only
symptom is a weak, incoherent contact map.

The WiLoR hand detector emits class 1 for a right hand and class 0 for a left hand, so a sparse
sample of frames gives an unambiguous tally. Confidence is summed as well as counted, so one crisp detection
outweighs a scatter of marginal ones.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

from lib.utils.hand_detector import HandDetector, LEFT, RIGHT
from lib.utils.imgseq_capture import open_capture

CONF = 0.3          # same detector confidence Stage A uses, so the tally reflects what Stage A sees
IOU = 0.3


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--video', required=True)
    ap.add_argument('--detector_path', default='data/base_data/demo_data/wilor_detector.onnx')
    ap.add_argument('--samples', type=int, default=48, help='frames to probe, spread over the video')
    ap.add_argument('--report', default='')
    args = ap.parse_args()

    detector = HandDetector(args.detector_path)

    cap = open_capture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:                      # some containers under-report; scan until the decoder stops
        total = 10 ** 7
    wanted = set(np.linspace(0, max(0, total - 1), num=max(1, args.samples)).astype(int).tolist())
    last = max(wanted)

    frames = {'right': 0, 'left': 0}    # frames containing at least one hand of that side
    conf = {'right': 0.0, 'left': 0.0}  # summed detection confidence
    seen = 0
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok or fi > last:
            break
        if fi in wanted:
            seen += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hit = {'right': False, 'left': False}
            for *_, c, cls in detector(rgb, conf=CONF, iou=IOU):
                side = 'right' if cls == RIGHT else ('left' if cls == LEFT else None)
                if side:
                    hit[side] = True
                    conf[side] += c
            for side in ('right', 'left'):
                frames[side] += int(hit[side])
        fi += 1
    cap.release()

    # Frame count decides; summed confidence breaks a tie. A 0-0 tally falls to right, which is
    # the model's native convention — and the log line below says so, because that case usually
    # means the detector found no hand at all rather than that the video is right-handed.
    if frames['left'] != frames['right']:
        hand = 'left' if frames['left'] > frames['right'] else 'right'
    else:
        hand = 'left' if conf['left'] > conf['right'] else 'right'

    result = {'hand': hand, 'frames_probed': seen,
              'right_frames': frames['right'], 'left_frames': frames['left'],
              'right_conf': round(conf['right'], 3), 'left_conf': round(conf['left'], 3),
              'video': os.path.abspath(args.video)}
    print(f"[probe] sampled {seen} frames: right={frames['right']} (conf {conf['right']:.1f}) "
          f"left={frames['left']} (conf {conf['left']:.1f}) -> {hand}", flush=True)
    if frames['right'] == 0 and frames['left'] == 0:
        print("[probe] WARNING: no hand detected in any sampled frame — defaulting to right. "
              "Stage A will still run, but if it never finds the hand it crops the whole frame, so "
              "the contact map will be poor. Check the video, or pass --hand explicitly.", flush=True)
    elif min(frames.values()) > 0.6 * max(frames.values()):
        print(f"[probe] NOTE: both hands appear about equally often "
              f"({frames['right']} right / {frames['left']} left). This pipeline tracks ONE hand; "
              f"pass --hand to choose, or run it twice.", flush=True)
    if args.report:
        with open(args.report, 'w') as fh:
            json.dump(result, fh, indent=2)
    print('PROBE_JSON ' + json.dumps(result), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
