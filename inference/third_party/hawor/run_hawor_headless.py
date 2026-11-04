#!/usr/bin/env python3
"""Run HaWoR's video pipeline on hand detections we supply, and export a frame-aligned NPZ.

HaWoR normally finds hands itself (scripts/scripts_test_video/detect_track_video.py, through
lib/pipeline/tools.py, the only HaWoR module that imports the Ultralytics library). This runner never
imports that module. It takes the hand detections Stage A already computed (--detections, our ONNX hand
detector) and writes the same on-disk inputs that HaWoR's own detection step would. Those are the frames
in {seq}/extracted_images/ and tracks_0_N/model_tracks.npy, a dict of hand tracks, each a list of
{'frame', 'det', 'det_box' [x1,y1,x2,y2,conf], 'det_handedness'}. Then it runs HaWoR's motion
estimation, SLAM and infiller unchanged. hawor_video.py needs two small helpers from
lib/pipeline/tools.py, so a stand-in module with our own versions of them is registered first. Any
attempt to import Ultralytics raises.
"""

from __future__ import annotations

import argparse
import gc
import importlib.abc
import shutil
import subprocess
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch


class _NoUltralytics(importlib.abc.MetaPathFinder):
    """Fail loudly if anything tries to import Ultralytics; this runner must not depend on it."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "ultralytics" or fullname.startswith("ultralytics."):
            raise ImportError(f"{fullname} must not be imported here (hand detections come from Stage A)")
        return None


sys.meta_path.insert(0, _NoUltralytics())


def _runs(frame) -> list[tuple[int, int]]:
    """[a, b) index ranges of the maximal runs of consecutive frame numbers."""
    n = len(frame)
    if n == 0:
        return []
    cuts = (np.flatnonzero(np.diff(np.asarray(frame)) != 1) + 1).tolist()
    bounds = [0, *cuts, n]
    return list(zip(bounds[:-1], bounds[1:]))


def _parse_chunks(frame, boxes, min_len=16):
    """Split a track into runs of consecutive frames at least `min_len` long -> (frames, boxes) lists."""
    keep = [(a, b) for a, b in _runs(frame) if b - a >= min_len]
    return [frame[a:b] for a, b in keep], [boxes[a:b] for a, b in keep]


def _parse_chunks_hand_frame(frame):
    """Split frame numbers into runs of consecutive frames."""
    return [frame[a:b] for a, b in _runs(frame)]


def _install_tools_stand_in() -> None:
    """Register `lib.pipeline.tools` as our module so HaWoR's hawor_video.py imports these helpers from
    it instead of loading HaWoR's own tools.py, which imports Ultralytics at module level."""
    import lib.pipeline  # HaWoR package; its __init__ is empty

    stand_in = types.ModuleType("lib.pipeline.tools")
    stand_in.parse_chunks = _parse_chunks
    stand_in.parse_chunks_hand_frame = _parse_chunks_hand_frame
    sys.modules["lib.pipeline.tools"] = stand_in
    lib.pipeline.tools = stand_in


_install_tools_stand_in()

from hawor.utils.process import get_mano_faces, run_mano  # noqa: E402
from lib.eval_utils.custom_utils import load_slam_cam  # noqa: E402
from scripts.scripts_test_video.hawor_slam import hawor_slam  # noqa: E402
from scripts.scripts_test_video.hawor_video import (  # noqa: E402
    hawor_infiller,
    hawor_motion_estimation,
)

DET_CONF = 0.2          # minimum detection confidence used for tracking
LINK_IOU = 0.3          # IoU needed to continue a track from its last box
LINK_MAX_GAP = 10       # frames a track may go unseen and still be continued
DEDUP_IOU = 0.7         # same-frame boxes overlapping more than this are one hand (keep the stronger)
MIN_TRACK_LEN = 3       # shorter tracks are dropped as spurious


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between (N,4) and (M,4) xyxy boxes -> (N,M)."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=-1)
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=-1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=-1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def link_tracks(frame, boxes, conf, cls, n_frames: int) -> dict:
    """Group per-frame detections into hand tracks, independent of their left/right class.

    The detector's left/right label flips on some frames, and its per-class suppression can box one hand
    twice (once per class), so detections are not grouped by class. Per frame, near-duplicate boxes are
    merged. Each box continues the live track (seen within LINK_MAX_GAP frames) it overlaps most, if by at
    least LINK_IOU, greedily by overlap; otherwise it starts a new track. Tracks shorter than MIN_TRACK_LEN
    are dropped. Each track then takes its majority hand, the rule HaWoR itself applies to tracks, and
    keeps one box per frame and hand, the most confident.
    Returns {track_id: [{'frame', 'det', 'det_box' (1,5) float32, 'det_handedness' (1,) float32}, ...]}.
    """
    frame = np.asarray(frame, np.int64).reshape(-1)
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    conf = np.asarray(conf, np.float32).reshape(-1)
    cls = np.asarray(cls, np.float32).reshape(-1)
    rows_kept, track_of = [], []
    last_box, last_seen = [], []
    for t in range(n_frames):
        rows = np.flatnonzero((frame == t) & (conf >= DET_CONF))
        rows = rows[np.argsort(-conf[rows], kind="stable")]
        if len(rows) > 1:
            ov = _iou(boxes[rows], boxes[rows])
            rows = rows[[not np.any(ov[i, :i] > DEDUP_IOU) for i in range(len(rows))]]
        live = [k for k in range(len(last_box)) if t - last_seen[k] <= LINK_MAX_GAP]
        assigned = {}
        if live and len(rows):
            ov = _iou(boxes[rows], np.stack([last_box[k] for k in live]))
            for flat in np.argsort(-ov, axis=None, kind="stable"):
                i, j = np.unravel_index(flat, ov.shape)
                if ov[i, j] < LINK_IOU:
                    break
                if i in assigned or live[j] in assigned.values():
                    continue
                assigned[i] = live[j]
        for i, r in enumerate(rows):
            k = assigned.get(i)
            if k is None:
                k = len(last_box)
                last_box.append(None)
                last_seen.append(None)
            last_box[k], last_seen[k] = boxes[r], t
            rows_kept.append(r)
            track_of.append(k)

    rows_kept, track_of = np.asarray(rows_kept, np.int64), np.asarray(track_of, np.int64)
    hand_of_track = {}
    for k in np.unique(track_of):
        mine = rows_kept[track_of == k]
        if len(mine) >= MIN_TRACK_LEN:
            hand_of_track[int(k)] = np.float32((cls[mine] > 0).mean() >= 0.5)

    # rows_kept is in frame order, most confident first within a frame; keep the first box of each hand
    # per frame, so HaWoR sees at most one right and one left hand per frame (as from its own detector)
    tracks: dict = {}
    taken = set()
    for r, k in zip(rows_kept.tolist(), track_of.tolist()):
        if k not in hand_of_track:
            continue
        hand = hand_of_track[k]
        if (int(frame[r]), float(hand)) in taken:
            continue
        taken.add((int(frame[r]), float(hand)))
        tracks.setdefault(k, []).append({
            "frame": int(frame[r]),
            "det": True,
            "det_box": np.concatenate([boxes[r], conf[r:r + 1]]).astype(np.float32)[None],
            "det_handedness": np.asarray([hand], np.float32),
        })
    return tracks


def prepare_sequence(video_path: Path, detections: Path):
    """Write what HaWoR's detection step would: the frames and tracks_0_N/{model_tracks,model_boxes}.npy.

    Frames are extracted one-to-one with the video's own frames (no resampling), so that frame i is the
    frame Stage A detected on. -> (start_idx, end_idx, seq_folder, image_paths)
    """
    seq_folder = video_path.parent / video_path.name.split(".")[0]
    img_folder = seq_folder / "extracted_images"
    img_folder.mkdir(parents=True, exist_ok=True)
    images = sorted(img_folder.glob("*.jpg"))
    if not images:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(video_path), "-fps_mode", "passthrough",
                        "-start_number", "0", str(img_folder / "%04d.jpg")], check=True)
        images = sorted(img_folder.glob("*.jpg"))
    n = len(images)

    with np.load(detections) as z:
        n_det = int(len(z["frame_idx"])) if "frame_idx" in z.files else None
        frame, boxes, conf, cls = z["det_frame"], z["det_xyxy"], z["det_conf"], z["det_cls"]
    if n_det is not None and n_det != n:
        raise RuntimeError(f"{detections} covers {n_det} frames, {video_path} has {n}")
    tracks = link_tracks(frame, boxes, conf, cls, n)
    print(f"[hawor] {len(frame)} detections -> {len(tracks)} hand tracks "
          f"{ {k: len(v) for k, v in tracks.items()} }", flush=True)

    track_dir = seq_folder / f"tracks_0_{n}"
    track_file = track_dir / "model_tracks.npy"
    if track_file.exists() and not _same_tracks(np.load(track_file, allow_pickle=True).item(), tracks):
        # HaWoR skips every stage whose output exists, so results from other detections must go
        print(f"[hawor] detections changed since the last run: clearing {seq_folder} (frames kept)", flush=True)
        for child in seq_folder.iterdir():
            if child.name != "extracted_images":
                shutil.rmtree(child) if child.is_dir() else child.unlink()
    track_dir.mkdir(parents=True, exist_ok=True)
    np.save(track_file, np.array(tracks, dtype=object))
    np.save(track_dir / "model_boxes.npy", np.array([], dtype=object))   # HaWoR's "detection done" marker
    return 0, n, str(seq_folder), [str(p) for p in images]


def _same_tracks(a: dict, b: dict) -> bool:
    if sorted(a) != sorted(b):
        return False
    for k in a:
        if len(a[k]) != len(b[k]):
            return False
        for x, y in zip(a[k], b[k]):
            if x["frame"] != y["frame"] or not np.array_equal(x["det_box"], y["det_box"]) \
                    or not np.array_equal(x["det_handedness"], y["det_handedness"]):
                return False
    return True


def wrist_cap(faces: np.ndarray) -> np.ndarray:
    """Faces sealing the MANO wrist opening (the mesh's one boundary loop), fanned from one rim vertex.

    A rim edge is a directed face edge whose reverse is in no face; each cap face runs over its rim edge in
    the reverse direction, so the closed mesh keeps the original winding (same rule as
    tools/make_mano_assets.py).
    """
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).tolist()
    present = set(map(tuple, edges))
    succ = {a: b for a, b in edges if (b, a) not in present}
    loop = [next(iter(succ))]
    while succ[loop[-1]] != loop[0]:
        loop.append(succ[loop[-1]])
    return np.array([[loop[0], loop[i + 1], loop[i]] for i in range(1, len(loop) - 1)], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--detections", required=True,
                        help="Stage A npz with the per-frame hand detections (det_frame, det_xyxy, det_conf, "
                             "det_cls) of this video")
    parser.add_argument(
        "--checkpoint",
        default="./weights/hawor/checkpoints/hawor.ckpt",
    )
    parser.add_argument(
        "--infiller-weight",
        default="./weights/hawor/checkpoints/infiller.pt",
    )
    parser.add_argument("--img-focal", type=float, default=600.0)
    parser.add_argument("--source-fps", type=float, default=30000.0 / 1001.0)
    parser.add_argument("--expected-frames", type=int, default=None)
    return parser.parse_args()


def _right_observations(
    tracks_path: Path, frame_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tracks = np.load(tracks_path, allow_pickle=True).item()
    observed = np.zeros(frame_count, dtype=bool)
    confidence = np.full(frame_count, np.nan, dtype=np.float32)
    boxes = np.full((frame_count, 4), np.nan, dtype=np.float32)

    for track in tracks.values():
        detections = [item for item in track if bool(item.get("det", False))]
        if not detections:
            continue
        handedness = np.concatenate(
            [np.asarray(item["det_handedness"]).reshape(-1) for item in detections]
        )
        if float(np.mean(handedness > 0)) < 0.5:
            continue
        for item in detections:
            frame = int(item["frame"])
            if frame < 0 or frame >= frame_count:
                continue
            box = np.asarray(item["det_box"], dtype=np.float32).reshape(-1, 5)[0]
            if not observed[frame] or box[4] > confidence[frame]:
                observed[frame] = True
                boxes[frame] = box[:4]
                confidence[frame] = box[4]
    return observed, confidence, boxes


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    args = parse_args()
    # Keep a benchmark symlink path intact so native HaWoR writes beside it.
    video_path = Path(args.video_path).absolute()
    output_path = Path(args.output).resolve()

    # The native functions use these attribute names.
    args.video_path = str(video_path)
    args.input_type = "file"

    start_idx, end_idx, seq_folder_str, image_paths = prepare_sequence(video_path, Path(args.detections))
    seq_folder = Path(seq_folder_str)
    frame_count = len(image_paths)
    if args.expected_frames is not None and frame_count != args.expected_frames:
        raise RuntimeError(
            f"expected {args.expected_frames} source frames, found {frame_count}"
        )
    if start_idx != 0 or end_idx != frame_count:
        raise RuntimeError(
            f"unexpected native interval [{start_idx}, {end_idx}) for {frame_count} frames"
        )

    tracks_path = (
        seq_folder / f"tracks_{start_idx}_{end_idx}" / "model_tracks.npy"
    )
    observed, confidence, bbox_xyxy = _right_observations(
        tracks_path, frame_count
    )

    frame_chunks, focal = hawor_motion_estimation(
        args, start_idx, end_idx, str(seq_folder)
    )
    slam_path = (
        seq_folder / "SLAM" / f"hawor_slam_w_scale_{start_idx}_{end_idx}.npz"
    )
    if not slam_path.exists():
        hawor_slam(args, start_idx, end_idx)
    if not slam_path.exists():
        raise RuntimeError(f"HaWoR did not create {slam_path}")

    pred_trans, pred_rot, pred_pose, pred_betas, filled = hawor_infiller(
        args, start_idx, end_idx, frame_chunks
    )
    if pred_trans.shape[:2] != (2, frame_count):
        raise RuntimeError(
            f"unexpected HaWoR output shape {tuple(pred_trans.shape)}"
        )

    gc.collect()
    torch.cuda.empty_cache()
    with torch.no_grad():
        mano = run_mano(
            pred_trans[1:2],
            pred_rot[1:2],
            pred_pose[1:2],
            betas=pred_betas[1:2],
        )
    verts_world = mano["vertices"][0].detach().cpu()
    joints_world = mano["joints"][0].detach().cpu()

    cam_R, cam_t, cam_R_inv, cam_t_inv = [
        value.float() for value in load_slam_cam(str(slam_path))
    ]
    if len(cam_R) != frame_count:
        raise RuntimeError(
            f"SLAM returned {len(cam_R)} cameras for {frame_count} frames"
        )
    verts_cam = (
        torch.einsum("tij,tnj->tni", cam_R, verts_world)
        + cam_t[:, None, :]
    )
    joints_cam = (
        torch.einsum("tij,tnj->tni", cam_R, joints_world)
        + cam_t[:, None, :]
    )

    first_image = cv2.imread(str(image_paths[0]))
    if first_image is None:
        raise RuntimeError(f"could not read {image_paths[0]}")
    height, width = first_image.shape[:2]
    with np.load(slam_path) as slam_data:
        center_value = (
            slam_data["img_center"]
            if "img_center" in slam_data.files
            else [width / 2.0, height / 2.0]
        )
        center = np.asarray(center_value, dtype=np.float32).reshape(2)
    K = np.array(
        [[focal, 0.0, center[0]], [0.0, focal, center[1]], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    mano_faces = np.asarray(get_mano_faces(), dtype=np.int64)
    faces = np.concatenate([mano_faces, wrist_cap(mano_faces)], axis=0)
    filled_right = np.asarray(filled[1]).astype(bool)
    render_valid = filled_right & np.all(
        np.isfinite(verts_cam.numpy()), axis=(1, 2)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema_version=np.asarray(1, dtype=np.int64),
        method=np.asarray("HaWoR"),
        hand_side=np.asarray("right"),
        tracker_commit=np.asarray(_git_commit(Path(__file__).resolve().parent)),
        source_video=np.asarray(str(video_path)),
        source_fps=np.asarray(args.source_fps, dtype=np.float64),
        frame_idx=np.arange(frame_count, dtype=np.int64),
        verts_cam=verts_cam.numpy().astype(np.float32),
        joints3d_cam=joints_cam.numpy().astype(np.float32),
        joint_convention=np.asarray("mano_openpose_21"),
        verts_world=verts_world.numpy().astype(np.float32),
        joints3d_world=joints_world.numpy().astype(np.float32),
        faces=faces,
        observed=observed,
        interpolated=render_valid & ~observed,
        render_valid=render_valid,
        confidence=confidence,
        bbox_xyxy=bbox_xyxy,
        K=K,
        img_wh=np.asarray([width, height], dtype=np.int64),
        mano_global_orient=pred_rot[1].numpy().astype(np.float32),
        mano_hand_pose=pred_pose[1].numpy().astype(np.float32),
        mano_betas=pred_betas[1].numpy().astype(np.float32),
        cam_R_w2c=cam_R.numpy().astype(np.float32),
        cam_t_w2c=cam_t.numpy().astype(np.float32),
        cam_R_c2w=cam_R_inv.numpy().astype(np.float32),
        cam_t_c2w=cam_t_inv.numpy().astype(np.float32),
    )
    print(
        f"wrote {output_path}: {int(observed.sum())}/{frame_count} observed, "
        f"{int((render_valid & ~observed).sum())} filled"
    )


if __name__ == "__main__":
    main()
