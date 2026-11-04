"""Smooth hand tracking for cop-viz and report what changed.

    uv run hand_tracking_process/smooth_hand_tracking.py                       # the full IMG_3970 clip
    uv run hand_tracking_process/smooth_hand_tracking.py --npz data/IMG_3970_c02_1316-1636_combined.npz
    uv run --with matplotlib hand_tracking_process/smooth_hand_tracking.py --plot
    uv run hand_tracking_process/smooth_hand_tracking.py --mode butterworth \
        --npz raw_hawor.npz --output copviz_2hz_track.npz --report copviz_2hz_velocity_report.json

The default ``sg`` mode is the legacy HACO_seq smoother described below. The ``butterworth`` mode is
the post-tracker cop-viz stage: it reads an immutable raw HaWoR archive, filters wrist translation at
2 Hz in the 3D world frame and hand articulation at 2 Hz in the wrist frame, and writes a separate
derived track for the HACO/copy-viz combiner.

Writes two files next to ``--out``:

  <TAG>_smooth_combined.npz   a DROP-IN combined npz (same schema, smoothed ``verts``/``cam_t``/``found``) —
                              feed it straight to run_cop_fit.py / bake_full_viz.py / the viewer
  <TAG>_hand_track.npz        the tracking proper: wrist R|t, fingertips in camera space AND in the wrist
                              frame, linear/angular velocity, and the valid/interpolated masks

The report is a before/after of the two things asked for — wrist pose smoothness, and fingertip smoothness
*in the wrist frame* — plus a FIDELITY column, because any filter can win a smoothness contest by deleting
the signal. Fidelity is how far the smoothed track actually moved from the raw one; it must stay small
against the hand (165 mm) or the "smoothing" is just distortion.
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as op
import sys
from datetime import datetime, timezone

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, op.dirname(op.abspath(__file__)))
from hand_smoother import (  # noqa: E402
    ButterworthCfg,
    ButterworthHandFilter,
    HandTrackingSmoother,
    SmoothCfg,
)

DEFAULT_NPZ = "out/example/example_vitl_combined.npz"
DEFAULT_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "base_data", "mano_assets.npz")
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


def jerk_mm(x):
    """Mean |2nd time-difference|, in mm — the high-frequency content of a track. Frame-rate independent
    enough to compare raw vs smoothed on the same clip, which is all we ask of it."""
    return 1e3 * float(np.abs(np.diff(np.asarray(x, float), 2, axis=0)).mean())


def step_mm(x):
    """Mean per-frame displacement, mm."""
    return 1e3 * float(np.linalg.norm(np.diff(np.asarray(x, float), axis=0), axis=-1).mean())


def rot_step_deg(R):
    """Mean rotation between consecutive frames, degrees."""
    rel = np.transpose(R[:-1], (0, 2, 1)) @ R[1:]
    return float(np.degrees(Rotation.from_matrix(rel).magnitude()).mean())


def rot_jerk_deg(R):
    """Angular jerk: mean |2nd difference| of the (unwrapped) rotation-vector track, in degrees."""
    rv = Rotation.from_matrix(R).as_rotvec()
    return float(np.degrees(np.abs(np.diff(rv, 2, axis=0))).mean())


def _required(z, key):
    if key not in z.files:
        raise ValueError(f"input archive is missing required key: {key}")
    return np.asarray(z[key])


def _scalar(z, *keys):
    for key in keys:
        if key in z.files:
            value = np.asarray(z[key])
            if value.size != 1:
                raise ValueError(f"{key} must be scalar")
            return value.reshape(-1)[0].item()
    raise ValueError(f"input archive needs one of: {', '.join(keys)}")


def _projection_centroid_displacement(raw, filtered, K):
    n = len(raw)
    K = np.asarray(K, np.float64)
    if K.shape == (3, 3):
        K = np.broadcast_to(K, (n, 3, 3))
    if K.shape != (n, 3, 3):
        raise ValueError(f"K must have shape (3, 3) or ({n}, 3, 3)")

    def centroid(points):
        homogeneous = np.einsum("tij,tnj->tni", K, points)
        pixels = homogeneous[..., :2] / homogeneous[..., 2:3]
        return pixels.mean(axis=1)

    displacement = np.linalg.norm(centroid(filtered) - centroid(raw), axis=1)
    return {
        "mean_px": float(displacement.mean()),
        "median_px": float(np.median(displacement)),
        "p95_px": float(np.percentile(displacement, 95)),
        "max_px": float(displacement.max()),
    }


def run_butterworth(args):
    if not args.output or not args.report:
        raise ValueError("butterworth mode requires --output and --report")
    source = op.abspath(op.expanduser(args.npz))
    output = op.abspath(op.expanduser(args.output))
    report_path = op.abspath(op.expanduser(args.report))
    if source == output:
        raise ValueError("--output must differ from the raw tracker input")

    with np.load(source, allow_pickle=False) as z:
        payload = {key: np.asarray(z[key]) for key in z.files}
        fps = float(args.fps if args.fps is not None else _scalar(z, "source_fps", "fps"))
        verts_cam = _required(z, "verts_cam")
        joints_cam = _required(z, "joints3d_cam")
        observed = np.asarray(
            z["observed"] if "observed" in z.files else _required(z, "render_valid"),
            bool,
        )
        interpolated = np.asarray(
            z["interpolated"] if "interpolated" in z.files else np.zeros(len(verts_cam), bool),
            bool,
        )
        result = ButterworthHandFilter(
            fps=fps,
            cfg=ButterworthCfg(
                cutoff_hz=args.cutoff_hz,
                prototype_order=args.order,
                max_interpolated_gap=args.max_interpolated_gap,
                minimum_run_frames=args.min_run_frames,
            ),
        ).run(
            verts_cam,
            joints_cam,
            observed,
            interpolated,
            _required(z, "cam_R_c2w"),
            _required(z, "cam_t_c2w"),
            _required(z, "cam_R_w2c"),
            _required(z, "cam_t_w2c"),
        )
        projection = _projection_centroid_displacement(
            verts_cam, result["verts_cam"], _required(z, "K")
        )
        source_world_residual = {}
        for key, calculated in (
            ("verts_world", ButterworthHandFilter._camera_to_world(
                np.asarray(verts_cam, np.float64),
                np.asarray(z["cam_R_c2w"], np.float64),
                np.asarray(z["cam_t_c2w"], np.float64),
            )),
            ("joints3d_world", ButterworthHandFilter._camera_to_world(
                np.asarray(joints_cam, np.float64),
                np.asarray(z["cam_R_c2w"], np.float64),
                np.asarray(z["cam_t_c2w"], np.float64),
            )),
        ):
            source_world_residual[key] = (
                None if key not in z.files
                else float(np.max(np.abs(np.asarray(z[key], np.float64) - calculated)))
            )

    filter_config = result["filter"]
    payload.update(
        verts_cam=result["verts_cam"].astype(np.float32),
        joints3d_cam=result["joints3d_cam"].astype(np.float32),
        verts_world=result["verts_world"].astype(np.float32),
        joints3d_world=result["joints3d_world"].astype(np.float32),
        cop_viz_filter_schema_version=np.asarray(1, np.int64),
        cop_viz_filter_owner=np.asarray("cop-viz"),
        cop_viz_filter_stage=np.asarray("hand_tracking_process"),
        cop_viz_filter_source_npz=np.asarray(source),
        cop_viz_filter_type=np.asarray(filter_config["type"]),
        cop_viz_filter_cutoff_hz=np.asarray(filter_config["cutoff_hz"], np.float64),
        cop_viz_filter_design_cutoff_hz=np.asarray(filter_config["design_cutoff_hz"], np.float64),
        cop_viz_filter_prototype_order=np.asarray(filter_config["prototype_order"], np.int64),
        cop_viz_filter_effective_order=np.asarray(filter_config["effective_zero_phase_order"], np.int64),
        cop_viz_filter_zero_phase=np.asarray(filter_config["zero_phase"]),
        cop_viz_filter_trusted_interpolated=result["trusted_interpolated"],
        cop_viz_filter_support=result["support"],
        cop_viz_filter_applied=result["applied"],
        cop_viz_filter_edge_sensitive=result["edge_sensitive"],
        cop_viz_filter_metric_mask=result["metric_mask"],
        cop_viz_filter_raw_wrist_world=result["raw_wrist_world"].astype(np.float32),
        cop_viz_filter_filtered_wrist_world=result["filtered_wrist_world"].astype(np.float32),
        cop_viz_filter_raw_fingertips_wrist=result["raw_fingertips_wrist"].astype(np.float32),
        cop_viz_filter_filtered_fingertips_wrist=result["filtered_fingertips_wrist"].astype(np.float32),
        cop_viz_filter_config_json=np.asarray(json.dumps(filter_config, sort_keys=True)),
        cop_viz_velocity_benchmark_json=np.asarray(
            json.dumps(result["velocity_benchmark"], sort_keys=True)
        ),
    )
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "owner": "cop-viz",
        "stage": "hand_tracking_process",
        "raw_tracker_input": source,
        "derived_track_output": output,
        "raw_tracker_method": str(np.asarray(payload.get("method", "unknown")).reshape(-1)[0]),
        "filter": filter_config,
        "velocity_benchmark": result["velocity_benchmark"],
        "validation": result["validation"] | {
            "source_world_residual_max_abs": source_world_residual,
            "render_centroid_displacement": projection,
        },
    }

    for path in (output, report_path):
        os.makedirs(op.dirname(path) or ".", exist_ok=True)
    temporary = f"{output}.tmp-{os.getpid()}.npz"
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, output)
    finally:
        if op.exists(temporary):
            os.unlink(temporary)
    temporary_report = f"{report_path}.tmp-{os.getpid()}"
    try:
        with open(temporary_report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_report, report_path)
    finally:
        if op.exists(temporary_report):
            os.unlink(temporary_report)

    wrist = result["velocity_benchmark"]["wrist_world"]
    tips = result["velocity_benchmark"]["fingertips_wrist"]
    print(
        f"WRIST WORLD velocity: {wrist['raw']['mean']:.4f} -> "
        f"{wrist['filtered']['mean']:.4f} m/s"
    )
    print(
        f"FINGERTIPS WRIST velocity: {tips['raw']['mean']:.4f} -> "
        f"{tips['filtered']['mean']:.4f} m/s"
    )
    print(f"[saved] {output} <- cop-viz derived track; raw tracker archive unchanged")
    print(f"[saved] {report_path} <- velocity benchmark")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("sg", "butterworth"), default="sg")
    ap.add_argument("--npz", default=DEFAULT_NPZ)
    ap.add_argument("--assets", default=DEFAULT_ASSETS)
    ap.add_argument("--out", default="out/hand_tracking")
    ap.add_argument("--output", help="explicit derived track path (butterworth mode)")
    ap.add_argument("--report", help="velocity benchmark JSON path (butterworth mode)")
    ap.add_argument("--cutoff-hz", type=float, default=2.0)
    ap.add_argument("--order", type=int, default=2, help="one-pass Butterworth prototype order")
    ap.add_argument("--fps", type=float, help="override archive source_fps")
    ap.add_argument("--min-run-frames", type=int)
    ap.add_argument("--max-interpolated-gap", type=int, default=3)
    ap.add_argument("--rot-mode", choices=["sg", "ema"], default="sg",
                    help="sg = zero-lag (default); ema = the old AriaHandsOptimizer's causal EMA")
    ap.add_argument("--sg-window", type=int, default=15,
                    help="in-plane wrist translation + wrist rotation window (default 15 ~= 2.3 Hz at "
                         "24 fps; was 7 ~= 5 Hz, which left the wrist visibly jittering). See SmoothCfg.")
    ap.add_argument("--sg-window-depth", type=int, default=15)
    ap.add_argument("--local-sg-window", type=int, default=9)
    ap.add_argument("--fill-max-gap", type=int, default=13)
    ap.add_argument("--jump-threshold", type=float, default=0.04)
    ap.add_argument("--jump-threshold-depth", type=float, default=0.50)
    ap.add_argument("--width", type=int, default=1280, help="source video width (principal point = W/2)")
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--plot", action="store_true", help="before/after traces (needs matplotlib)")
    args = ap.parse_args()

    if args.mode == "butterworth":
        try:
            run_butterworth(args)
        except (OSError, ValueError) as exc:
            ap.error(str(exc))
        return

    z = np.load(args.npz)
    assets = np.load(args.assets)
    tag = op.basename(args.npz).replace("_combined.npz", "")
    fps = float(z["fps"])
    hand_side = str(z["hand_side"]) if "hand_side" in z.files else "right"
    vid_W = int(z["img_wh"][0]) if "img_wh" in z.files else args.width
    vid_H = int(z["img_wh"][1]) if "img_wh" in z.files else args.height
    cfg = SmoothCfg(sg_window=args.sg_window, sg_window_depth=args.sg_window_depth,
                    local_sg_window=args.local_sg_window, fill_max_gap=args.fill_max_gap,
                    jump_threshold=args.jump_threshold, jump_threshold_depth=args.jump_threshold_depth,
                    rot_mode=args.rot_mode)

    sm = HandTrackingSmoother(assets, cfg, fps=fps, hand_side=hand_side)
    r = sm.run(z["verts"], z["cam_t"], z["found"], z["focal"], cx=vid_W / 2, cy=vid_H / 2)

    print("=" * 100)
    print(f"HAND-TRACKING SMOOTHING — {tag}  ({len(z['verts'])} frames @ {fps:g} fps, rot_mode={cfg.rot_mode})")
    print("=" * 100)
    for line in r["log"]:
        print(f"  {line}")

    # ---- wrist pose ---------------------------------------------------------------------------------
    tr, ts = r["wrist_t_raw"], r["wrist_t"]
    Rr, Rs = r["wrist_R_raw"], r["wrist_R"]
    print("\n  WRIST POSE" + " " * 26 + "raw        smoothed      fidelity (how far we moved it)")
    print("  " + "-" * 92)
    for i, ax in enumerate("XYZ"):
        note = "  <- depth: ill-conditioned, own window" if i == cfg.depth_axis else ""
        print(f"    position {ax}   jerk (mm)        {jerk_mm(tr[:, i:i+1]):8.2f}   {jerk_mm(ts[:, i:i+1]):8.2f}"
              f"      RMS {1e3 * float(np.sqrt(np.mean((ts[:, i] - tr[:, i]) ** 2))):7.1f} mm{note}")
    print(f"    orientation   step (deg/frame) {rot_step_deg(Rr):8.2f}   {rot_step_deg(Rs):8.2f}")
    print(f"                  jerk (deg)       {rot_jerk_deg(Rr):8.2f}   {rot_jerk_deg(Rs):8.2f}"
          f"      RMS {np.degrees(Rotation.from_matrix(np.transpose(Rr, (0, 2, 1)) @ Rs).magnitude()).mean():7.2f} deg")

    # ---- fingertips: world vs WRIST FRAME -----------------------------------------------------------
    kr, ks = r["kp21_raw"][:, [4, 8, 12, 16, 20]], r["kp21"][:, [4, 8, 12, 16, 20]]
    lr, ls = r["tips_local_raw"], r["tips_local"]
    print("\n  FINGERTIPS" + " " * 22 + "raw        smoothed     reduction")
    print("  " + "-" * 92)
    print(f"    camera space      jerk (mm)    {jerk_mm(kr):8.2f}   {jerk_mm(ks):8.2f}"
          f"     {100 * (1 - jerk_mm(ks) / jerk_mm(kr)):5.0f}%")
    print(f"    WRIST FRAME       jerk (mm)    {jerk_mm(lr):8.2f}   {jerk_mm(ls):8.2f}"
          f"     {100 * (1 - jerk_mm(ls) / jerk_mm(lr)):5.0f}%   <- the one that matters for grasp")
    print(f"\n    {'finger':8s} {'wrist-frame jerk raw':>21s} {'smoothed':>10s} {'reduction':>10s}   "
          f"{'fidelity':>9s}")
    for k, f in enumerate(FINGERS):
        a, b = jerk_mm(lr[:, k]), jerk_mm(ls[:, k])
        fid = 1e3 * float(np.sqrt(np.mean(np.sum((ls[:, k] - lr[:, k]) ** 2, -1))))
        print(f"    {f:8s} {a:19.2f}mm {b:9.2f}mm {100 * (1 - b / a):9.0f}%   {fid:7.1f}mm")

    # ---- REPROJECTION QC: did the hand stay on the video? --------------------------------------------
    # The one check that matters for a monocular tracker, and the one that caught a real bug in this script
    # (smoothing metric X,Y,Z independently shears the X-Z coupling that holds u = f·X/Z still, and walks the
    # mesh off the hand). Split by whether the gate repaired the frame: on a REPAIRED frame the raw pose was a
    # detector glitch, so a large deviation from it is the point, not an error.
    b = z["bbox_xyxy"].astype(float)
    W_, H_ = vid_W, vid_H
    def project(V, ct):
        P = V + ct[:, None, :]
        f = np.asarray(z["focal"], float)[:, None]
        return f * P[:, :, 0] / P[:, :, 2] + W_ / 2, f * P[:, :, 1] / P[:, :, 2] + H_ / 2
    u0, v0 = project(np.asarray(z["verts"], float), np.asarray(z["cam_t"], float))
    u1, v1 = project(r["verts"], r["cam_t"])
    def in_box(u, v):
        return ((u >= b[:, 0:1]) & (u <= b[:, 2:3]) & (v >= b[:, 1:2]) & (v <= b[:, 3:4])).mean(1)
    shift = np.sqrt((u1 - u0) ** 2 + (v1 - v0) ** 2).mean(1)
    i0, i1 = in_box(u0, v0), in_box(u1, v1)
    unt, rej = r["untouched"], r["rejected"]
    print("\n  REPROJECTION QC — is the smoothed mesh still on the hand in the video?")
    print("  " + "-" * 92)
    print(f"    frames the gate KEPT     (n={int(unt.sum())}): mesh-in-detection-box {100 * i0[unt].mean():.1f}%"
          f" -> {100 * i1[unt].mean():.1f}%   shift median {np.median(shift[unt]):.2f} px, "
          f"p95 {np.percentile(shift[unt], 95):.1f} px")
    if rej.any():
        print(f"    frames the gate REPAIRED (n={int(rej.sum())}): shift median "
              f"{np.median(shift[rej]):.0f} px — these were detector glitches (1-2 frame depth collapses); "
              f"moving away from them IS the repair")
    assert np.median(shift[unt]) < 5.0, "the smoothed mesh walked off the hand on untouched frames"

    # ---- did we destroy the signal? -----------------------------------------------------------------
    print(f"\n  FIDELITY: fingertip travel per frame: raw {step_mm(kr):.2f} mm -> smoothed {step_mm(ks):.2f} mm"
          f" — the motion is kept, the jitter is not."
          f"\n            (a 3D RMS-vs-raw number would be meaningless here: it is dominated by depth, the one"
          f"\n             axis the camera never really measured. The pixel shift above is the honest one.)")

    # ---- write --------------------------------------------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    keep = {k: z[k] for k in z.files if k not in ("verts", "cam_t", "found")}
    p1 = op.join(args.out, f"{tag}_smooth_combined.npz")
    np.savez_compressed(p1, verts=r["verts"].astype(np.float32), cam_t=r["cam_t"].astype(np.float32),
                        found=r["found"], **keep)
    p2 = op.join(args.out, f"{tag}_hand_track.npz")
    np.savez_compressed(
        p2, wrist_R=r["wrist_R"].astype(np.float32), wrist_t=r["wrist_t"].astype(np.float32),
        wrist_R_raw=r["wrist_R_raw"].astype(np.float32), wrist_t_raw=r["wrist_t_raw"].astype(np.float32),
        kp21=r["kp21"].astype(np.float32), kp21_raw=r["kp21_raw"].astype(np.float32),
        tips_local=r["tips_local"].astype(np.float32), tips_local_raw=r["tips_local_raw"].astype(np.float32),
        lin_vel=r["lin_vel"].astype(np.float32), ang_vel=r["ang_vel"].astype(np.float32),
        valid_in=r["valid_in"], valid_out=r["valid_out"], finger_names=np.array(FINGERS),
        frame_idx=z["frame_idx"], fps=z["fps"], cfg=np.array(json.dumps(cfg.__dict__)),
    )
    print(f"\n[saved] {p1}   ({op.getsize(p1) / 1e6:.1f} MB)  <- drop-in combined npz")
    print(f"[saved] {p2}   ({op.getsize(p2) / 1e6:.1f} MB)  <- wrist pose + fingertips (world + wrist frame)")

    if args.plot:
        plot(r, tag, args.out, cfg)


def plot(r, tag, out, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(r["wrist_t"])
    a, b = 0, min(n, 600)
    x = np.arange(a, b)
    fig, ax = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    # X/Y span ~700 mm while Z wanders over ~8 m — on one axis the in-plane track is a flat line. Twin axis.
    for i, lab in enumerate("XY"):
        ax[0].plot(x, 1e3 * r["wrist_t_raw"][a:b, i], lw=.7, alpha=.45, color=f"C{i}")
        ax[0].plot(x, 1e3 * r["wrist_t"][a:b, i], lw=1.6, color=f"C{i}", label=f"{lab} (left axis)")
    axz = ax[0].twinx()
    axz.plot(x, 1e3 * r["wrist_t_raw"][a:b, 2], lw=.7, alpha=.45, color="C2")
    axz.plot(x, 1e3 * r["wrist_t"][a:b, 2], lw=1.6, color="C2", label="Z / depth (right axis)")
    axz.set_ylabel("depth Z (mm)", color="C2")
    axz.legend(loc="upper right", fontsize=8)
    ax[0].set_title(f"{tag} — wrist position (faint = raw, bold = smoothed).  X/Y are well conditioned; "
                    f"Z is inferred from apparent size — own {cfg.sg_window_depth}-frame window")
    ax[0].set_ylabel("in-plane X,Y (mm)")
    ax[0].legend(loc="upper left", ncol=2, fontsize=8)

    rv_r = Rotation.from_matrix(r["wrist_R_raw"]).as_rotvec()
    rv_s = Rotation.from_matrix(r["wrist_R"]).as_rotvec()
    for i in range(3):
        ax[1].plot(x, np.degrees(rv_r[a:b, i]), lw=.7, alpha=.45, color=f"C{i}")
        ax[1].plot(x, np.degrees(rv_s[a:b, i]), lw=1.6, color=f"C{i}")
    ax[1].set_title("wrist orientation (rotation vector)")
    ax[1].set_ylabel("deg")

    for k, f in enumerate(FINGERS):
        ax[2].plot(x, 1e3 * np.linalg.norm(r["kp21_raw"][a:b, [4, 8, 12, 16, 20][k]], axis=-1),
                   lw=.7, alpha=.4, color=f"C{k}")
        ax[2].plot(x, 1e3 * np.linalg.norm(r["kp21"][a:b, [4, 8, 12, 16, 20][k]], axis=-1),
                   lw=1.3, color=f"C{k}", label=f)
    ax[2].set_title("fingertips — CAMERA space (|position|): dominated by wrist motion + depth noise")
    ax[2].set_ylabel("mm")
    ax[2].legend(ncol=5, fontsize=8)

    for k, f in enumerate(FINGERS):
        ax[3].plot(x, 1e3 * np.linalg.norm(r["tips_local_raw"][a:b, k], axis=-1), lw=.7, alpha=.4,
                   color=f"C{k}")
        ax[3].plot(x, 1e3 * np.linalg.norm(r["tips_local"][a:b, k], axis=-1), lw=1.3, color=f"C{k}", label=f)
    ax[3].set_title("fingertips — WRIST FRAME (|position|): articulation only. THIS is what we stabilize")
    ax[3].set_ylabel("mm")
    ax[3].set_xlabel("frame")
    ax[3].legend(ncol=5, fontsize=8)

    fig.tight_layout()
    p = op.join(out, f"{tag}_smoothing.png")
    fig.savefig(p, dpi=110)
    print(f"[saved] {p}")


if __name__ == "__main__":
    main()
