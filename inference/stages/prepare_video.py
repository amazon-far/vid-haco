#!/usr/bin/env python3
"""
STAGE 0 — normalise the user's video into the one form the rest of the pipeline assumes.

Three separate consumers index this video by frame number and must agree exactly: HaWoR
(`--expected-frames`), Stage A/B (absolute `frame_idx`) and the renderer (`open_capture` +
`frame_idx`). A phone video with variable frame timing, or a container whose reported frame
count disagrees with what a decoder actually yields, silently breaks that agreement. So we
re-stamp timestamps to a constant rate WITHOUT dropping or duplicating a single frame:

    -vf setpts=N/(FPS*TB)  -fps_mode passthrough

`setpts` rewrites presentation timestamps from the frame ORDINAL (N), and `passthrough`
forbids ffmpeg from inventing or discarding frames to hit the rate. The result has exactly
as many frames as the input, now at a constant FPS.

The rate must be an EXACT RATIONAL, and this is not a detail. If the output's `avg_frame_rate`
and `r_frame_rate` disagree — which is what happens when you re-stamp a 30/1 video to its
measured average of 5955/199 — then `ffmpeg -i v.mp4 out/%06d.jpg` resamples to the nominal
rate and writes ONE MORE image than the stream has frames, while ffprobe and OpenCV still see
the true count. The tracker extracts frames exactly that way and the rest of the pipeline uses
OpenCV, so the two would be off by a frame, silently, for the whole clip. We therefore pick a
rational rate, prefer the container's nominal rate when it is close to the measured average,
and verify after encoding that the two rates came out identical.

Left hands are mirrored here (`hflip`). The contact models are right-canonical — training
flips every left hand so the crop looks like a right hand — so a mirrored video is exactly
what the model saw in training, while a native left crop is out of distribution. HaWoR is
right-hand-only for the same reason. Output frames stay in mirror space, which is what the
render shows.

Trimming to --start-frame/--end-frame also happens here, so the prepared video always starts
at frame 0 and the pipeline never has to carry an offset between the tracker and the model.
"""
import argparse
import json
import os
import subprocess
import sys
from fractions import Fraction


def _rational(value):
    """Parse ffprobe's 'num/den' rate. Returns Fraction(0) for the '0/0' it uses for unknown."""
    try:
        num, den = (str(value).split('/') + ['1'])[:2]
        return Fraction(int(num), int(den)) if int(den) else Fraction(0)
    except (ValueError, ZeroDivisionError):
        return Fraction(0)


def ffprobe(video):
    """Container-level stream facts: dimensions, both frame rates (exact), frame count."""
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams',
         '-show_format', '-of', 'json', video],
        check=True, capture_output=True, text=True).stdout
    info = json.loads(out)
    if not info.get('streams'):
        raise SystemExit(f"[prep] no video stream in {video}")
    s = info['streams'][0]
    avg, r = _rational(s.get('avg_frame_rate')), _rational(s.get('r_frame_rate'))
    n = s.get('nb_frames')
    return {
        'width': int(s['width']), 'height': int(s['height']),
        'avg_rate': avg, 'r_rate': r, 'fps': float(avg or r),
        'nb_frames': int(n) if n and n.isdigit() else 0,
        'duration': float(info.get('format', {}).get('duration', 0.0) or 0.0),
    }


def choose_rate(src, requested):
    """Pick the exact rational frame rate to re-stamp to.

    Prefer the container's NOMINAL rate (`r_frame_rate`) when it is within 2% of the measured
    average: it is the rate the footage was shot at, it is usually a clean rational like 30/1 or
    30000/1001, and re-stamping to it makes avg == r so no decoder resamples. Fall back to the
    measured average when the two are far apart, which means the source is genuinely variable and
    the nominal rate is not describing it.
    """
    if requested:
        return Fraction(requested).limit_denominator(100000)
    avg, r = src['avg_rate'], src['r_rate']
    if avg and r and abs(float(r) - float(avg)) <= 0.02 * float(avg):
        return r
    return (avg or r).limit_denominator(100000)


def count_frames(video):
    """Decode-accurate frame count. The container's nb_frames is a hint, not a promise, and
    every downstream stage indexes by DECODED ordinal — so we count what a decoder yields."""
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
         '-show_entries', 'stream=nb_read_frames', '-of', 'csv=p=0', video],
        check=True, capture_output=True, text=True).stdout.strip()
    return int(out.split(',')[0]) if out and out.split(',')[0].isdigit() else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--video', required=True, help='user input video (any container ffmpeg reads)')
    ap.add_argument('--out', required=True, help='normalised constant-frame-rate mp4')
    ap.add_argument('--hand', default='right', choices=['right', 'left'],
                    help='left => mirror the video so the hand reads as right (model is right-canonical)')
    ap.add_argument('--fps', default='',
                    help="output constant frame rate, e.g. 30 or 30000/1001; "
                         "empty = the source's nominal rate (falling back to its measured average)")
    ap.add_argument('--start-frame', type=int, default=0, help='inclusive first source frame to keep')
    ap.add_argument('--end-frame', type=int, default=-1, help='inclusive last source frame; -1 = to the end')
    ap.add_argument('--crf', type=int, default=16, help='x264 quality for the prepared copy (lower = better)')
    ap.add_argument('--report', default='', help='write a JSON summary here')
    args = ap.parse_args()

    if not os.path.isfile(args.video):
        raise SystemExit(f"[prep] input video not found: {args.video}")
    src = ffprobe(args.video)
    rate = choose_rate(src, args.fps.strip())
    if rate <= 0:
        raise SystemExit(f"[prep] could not determine a frame rate for {args.video}; pass --fps")
    fps = float(rate)
    print(f"[prep] source {src['width']}x{src['height']} nominal={src['r_rate']} "
          f"measured={src['avg_rate']} ({float(src['avg_rate'] or 0):.4f} fps) "
          f"nb_frames={src['nb_frames']} dur={src['duration']:.2f}s", flush=True)
    print(f"[prep] re-stamping to exactly {rate} = {fps:.6f} fps", flush=True)

    # Frame-exact trimming: `select` keeps chosen ordinals, then setpts re-stamps the KEPT
    # frames from 0. Doing it with -ss/-t instead would cut on timestamps and land off by a
    # frame or two, which would shift every downstream frame_idx.
    filters = []
    if args.hand == 'left':
        filters.append('hflip')
    start, end = max(0, args.start_frame), args.end_frame
    if start > 0 or end >= 0:
        expr = f"gte(n\\,{start})" if end < 0 else f"between(n\\,{start}\\,{end})"
        filters.append(f"select='{expr}'")
    filters.append(f'setpts=N/(({rate.numerator}/{rate.denominator})*TB)')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', args.video,
           '-vf', ','.join(filters), '-fps_mode', 'passthrough',
           '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', str(args.crf),
           '-pix_fmt', 'yuv420p', args.out]
    print(f"[prep] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    n = count_frames(args.out)
    if n <= 0:
        raise SystemExit(f"[prep] prepared video has no decodable frames: {args.out}")
    out_info = ffprobe(args.out)

    # The check that stops a one-frame desync from reaching the rest of the pipeline. If the two
    # advertised rates differ, `ffmpeg -i out.mp4 %06d.jpg` (how the tracker reads frames) will
    # resample and disagree with OpenCV (how everything else reads them).
    if out_info['avg_rate'] != out_info['r_rate']:
        raise SystemExit(
            f"[prep] {args.out} advertises avg_frame_rate={out_info['avg_rate']} but "
            f"r_frame_rate={out_info['r_rate']}. Frame-based extraction would resample and "
            f"disagree with OpenCV by a frame or more. Re-run with an explicit rational --fps "
            f"(e.g. --fps {src['r_rate']}).")
    print(f"[prep] wrote {args.out}: {n} frames, {out_info['width']}x{out_info['height']} "
          f"@ {out_info['avg_rate']} (avg == r, decoders agree), mirrored={args.hand == 'left'}",
          flush=True)

    report = {'input': os.path.abspath(args.video), 'output': os.path.abspath(args.out),
              'hand': args.hand, 'mirrored': args.hand == 'left', 'fps': fps,
              'fps_rational': f"{rate.numerator}/{rate.denominator}",
              'n_frames': n, 'width': out_info['width'], 'height': out_info['height'],
              'start_frame': start, 'end_frame': end,
              'source': {k: (str(v) if isinstance(v, Fraction) else v) for k, v in src.items()}}
    if args.report:
        with open(args.report, 'w') as fh:
            json.dump(report, fh, indent=2)
    print('PREPARE_JSON ' + json.dumps(report), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
