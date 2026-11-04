"""Temporal smoothing for WiLoR / HACO_seq hand tracking — wrist pose + fingertips.

Ported in spirit from this repo's ``AriaHandsOptimizer.py`` (same stage order, same knobs, same defaults
where they still apply): reject jumps -> fill nulls -> Savitzky-Golay the positions -> smooth the rotation
-> derive velocities. Two things are DIFFERENT, and both are forced by measurements on this data:

1. FINGERTIPS ARE SMOOTHED IN THE WRIST FRAME, NOT IN WORLD.
   The reference SG-filters ``thumb/index_translation_raw_world`` in world space. A fingertip's world
   trajectory is (gross wrist motion) + (articulation), and those live at completely different frequencies —
   filtering their sum means every bit of residual wrist jitter is injected straight into the fingertips, and
   a fast wrist sweep gets flattened along with the noise. We instead decompose

       L = R_wrist^T (p_world - t_wrist)        # articulation only, in the wrist frame

   smooth L, smooth (R_wrist, t_wrist) separately, and recompose. The fingertips then come out smooth AND
   stable *relative to the hand*, which is what a downstream grasp/contact consumer actually needs.

   This is done on all 778 MANO vertices at once rather than on 5 tips, and that is free: kp21 is a LINEAR
   function of the vertices (J_regressor @ V, plus 5 tip vertices picked out), and Savitzky-Golay is a linear
   time filter, so smoothing the vertices and then regressing the joints is IDENTICAL to smoothing the joints
   — while also handing back a smoothed mesh the contact pipeline can reuse.

2. THE TRANSLATION IS SMOOTHED IN (u, v, Z), NOT IN (X, Y, Z).
   The reference gates and filters metric XYZ isotropically. That is wrong for a monocular tracker, and
   measurably so. WiLoR's cam_t has a median |dZ| of 99 mm/frame (p95 481 mm; Z ranges over 15 m across the
   clip — for a hand on a table) against 1.5 / 1.2 mm for X / Y, because z ~ 2f/(bbox·s) is ill-conditioned at
   focal 25000. But X and Z are NOT independent: the camera only ever observed the projection u = f·X/Z, so
   X and Z jitter TOGETHER, coherently, and u stays put. Filter them on separate windows and you shear that
   coupling apart — we measured it: independent per-axis SG dropped mesh-in-detection-box from 98.7% to 97.0%
   and shifted the reprojection by 11.7 px mean (one frame by 1554 px). It "smooths" the hand off the video.

   So we smooth what the camera actually measured: the pixel track (u, v) — well conditioned, short window —
   and the depth Z — ill conditioned, long ``sg_window_depth`` — then rebuild X = (u-cx)·Z/f. The 2D
   alignment is preserved by construction, while the depth (and therefore the apparent SIZE, which is the
   only thing depth is inferred from) gets denoised. Jump rejection follows the same logic: it gates on
   in-plane motion converted to metres at the nominal depth (so ``jump_threshold`` keeps the reference's
   physical meaning of "metres of hand travel per frame") plus a separate, looser gate on depth.
   Depth remains the weak axis afterwards; smoothing cannot create information the estimator never had.

Missing frames (``found=False``, and anything the jump gate drops) are filled the way the reference does it:
linear on translation, SLERP on rotation, linear on the wrist-frame articulation. Gaps up to ``fill_max_gap``
are treated as trustworthy; longer internal gaps are still filled (an SG filter needs a continuous signal)
but reported and flagged so downstream can refuse them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, savgol_filter, sosfiltfilt
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class SmoothCfg:
    """Defaults mirror ``AriaHandsOptimizer.__init__`` where the quantity still means the same thing."""

    sg_window: int = 15                   # reference: smooth_window (was 7) — RAISED 2026-07-25. 7 puts the
                                          # -3 dB corner near 5 Hz at 24 fps, so most tracker jitter (2-5 Hz)
                                          # survived and the wrist visibly wobbled. 15 ~= 2.3 Hz and cuts
                                          # mesh screen-space jitter from 2.54 to 1.35 px/vertex on HaWoR
                                          # IMG_4009 (raw 6.30). It is also the widest window that still
                                          # passes the fidelity assert in smooth_hand_tracking.py (<5 px
                                          # median shift on gate-untouched frames): measured worst case
                                          # 4.36 px over all 36 existing WiLoR IMG_4009 + IMG_3970 clips,
                                          # while 21 trips it at 5.28. Governs in-plane wrist translation
                                          # AND wrist rotation; depth and articulation have their own
                                          # windows below.
    sg_polyorder: int = 2                 # reference: smooth_polyorder
    sg_window_depth: int = 15             # NEW — depth is noise-dominated; see README for the measured trade-off
    local_sg_window: int = 9              # NEW — articulation (wrist-frame) window
    min_valid_frames: int = 5             # reference: smooth_min_valid_frames
    fill_max_gap: int = 13                # TUNED on this data (validate_inpaint.py): the reference's 3 was
                                          # far too conservative. Worst-case inpaint error at gap<=10 is 7.1 mm,
                                          # under MANO's 7.89 mm vertex spacing -> the vertex-locked CoP lands on
                                          # the same/adjacent vertex, invisible downstream. Recovers 83% of gap
                                          # frames (98% of gaps) vs 65% at 3. Gaps 15-34 (0.6-1.4s) stay flagged.
    jump_threshold: float = 0.04          # reference: smooth_jump_threshold — now on the IMAGE-PLANE part
    jump_threshold_depth: float = 0.50    # NEW — separate gate for the ill-conditioned depth axis
    jump_reanchor_after: int = 3          # reference: smooth_jump_reanchor_after
    ema_alpha: float = 0.3                # reference: smooth_ema_alpha_x / _y (used only when rot_mode="ema")
    rot_mode: str = "sg"                  # "sg" = zero-lag (default) | "ema" = the reference's causal EMA
    depth_axis: int = 2


# ----------------------------------------------------------------------------------------------------
# primitives
# ----------------------------------------------------------------------------------------------------
def _fit_window(win: int, n: int, poly: int) -> tuple[int, int]:
    """Clamp an SG window to the data: odd, <= n, > polyorder. Returns (window, polyorder), or (0,0) if the
    segment is too short to filter at all."""
    w = int(win) | 1                                        # force odd
    w = min(w, n if n % 2 == 1 else n - 1)
    if w < 3:
        return 0, 0
    return w, int(np.clip(poly, 1, w - 1))


def _sg(x: np.ndarray, win: int, poly: int) -> np.ndarray:
    """Zero-phase Savitzky-Golay along axis 0 (centred: no lag — legal because we smooth offline)."""
    w, p = _fit_window(win, len(x), poly)
    if not w:
        return x.copy()
    return savgol_filter(x, window_length=w, polyorder=p, axis=0, mode="interp")


def weighted_kabsch(P: np.ndarray, Q: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Rotation taking point set ``P`` onto ``Q`` (both (n,3)), weighted by ``w`` — the classic SVD solution,
    with the reflection case handled. Used to recover the wrist orientation from the mesh."""
    wn = w / (w.sum() + 1e-12)
    Pc = (wn[:, None] * P).sum(0)
    Qc = (wn[:, None] * Q).sum(0)
    H = (wn[:, None] * (P - Pc)).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def _ema_unit_vec(v, v_ema, alpha):
    """VERBATIM behaviour of AriaHandsOptimizer._ema_unit_vec: EMA a unit vector, flipping sign first so a
    180-degree ambiguity can't be averaged into a tumble."""
    v = np.asarray(v, float)
    v = v / (np.linalg.norm(v) + 1e-6)
    if v_ema is None:
        return v, v.copy()
    if float(v @ v_ema) < 0.0:
        v = -v
    vn = (1.0 - alpha) * v_ema + alpha * v
    vn /= np.linalg.norm(vn) + 1e-6
    return vn, vn.copy()


def _smooth_rot_ema(R: np.ndarray, alpha: float) -> np.ndarray:
    """The reference's rotation smoother: EMA the x/y basis vectors, then Gram-Schmidt back to SO(3).
    CAUSAL — it lags by roughly (1-alpha)/alpha frames. Kept for parity with the old pipeline."""
    out = np.empty_like(R)
    ex = ey = None
    for i in range(len(R)):
        x, ex = _ema_unit_vec(R[i][:, 0], ex, alpha)
        y, ey = _ema_unit_vec(R[i][:, 1], ey, alpha)
        z = np.cross(x, y)
        z /= np.linalg.norm(z) + 1e-6
        y = np.cross(z, x)
        out[i] = np.column_stack([x, y, z])
    return out


def _smooth_rot_sg(R: np.ndarray, win: int, poly: int) -> np.ndarray:
    """Zero-lag rotation smoothing: SG the quaternion track (sign-unwrapped so q and -q don't fight), then
    renormalize. Same filter as the positions, so rotation and translation stay in phase — an EMA on one and
    an SG on the other would shear them apart by the EMA's lag."""
    q = Rotation.from_matrix(R).as_quat()
    for i in range(1, len(q)):                              # sign-unwrap: keep the track continuous on S^3
        if q[i] @ q[i - 1] < 0:
            q[i] = -q[i]
    qs = _sg(q, win, poly)
    qs /= np.linalg.norm(qs, axis=1, keepdims=True) + 1e-12
    return Rotation.from_quat(qs).as_matrix()


def _interp_invalid(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Linearly interpolate the invalid rows of an (N,d) track from the valid ones; hold at the ends.
    (np.interp clamps outside the sample range, which is exactly the hold-nearest we want.)"""
    x = np.asarray(x, float).copy()
    idx = np.flatnonzero(valid)
    if len(idx) < 2:
        return x
    g = np.arange(len(x))
    for d in range(x.shape[1]):
        x[:, d] = np.interp(g, idx, x[idx, d])
    return x


def _segments(valid: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [start, end) runs of True — the reference's _extract_segments."""
    segs, i, n = [], 0, len(valid)
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i + 1
        while j < n and valid[j]:
            j += 1
        segs.append((i, j))
        i = j
    return segs


# ----------------------------------------------------------------------------------------------------
class HandTrackingSmoother:
    """Smooth a WiLoR hand track: wrist pose + per-vertex articulation, missing frames filled.

    ``verts`` (N,778,3) root-relative MANO, ``cam_t`` (N,3), ``found`` (N,) — i.e. exactly what a HACO_seq
    combined npz stores. ``assets`` is ``mano_assets.npz`` (J_regressor, lbs_weights, v_template, tip_vids,
    kp21_map). Everything is metres, camera space = ``verts + cam_t``.
    """

    def __init__(self, assets, cfg: SmoothCfg | None = None, fps: float = 24.0, hand_side: str = "right"):
        self.cfg = cfg or SmoothCfg()
        self.dt = 1.0 / float(fps)
        self.hand_side = hand_side
        self.J = np.asarray(assets["J_regressor"], np.float64)          # (16,778)
        self.W_wrist = np.asarray(assets["lbs_weights"], np.float64)[:, 0]   # skin weight to the wrist joint
        self.v_rest = np.asarray(assets["v_template"], np.float64)
        if hand_side == "left":                                        # mirror the RIGHT rest template so the
            self.v_rest = self.v_rest.copy()                           # wrist Kabsch (weighted_kabsch) stays a
            self.v_rest[:, 0] *= -1.0                                  # PROPER rotation for x-mirrored left verts
        self.tips = np.asarray(assets["tip_vids"], np.int64)
        self.kp_map = np.asarray(assets["kp21_map"], np.int64)
        self.log: list[str] = []

    # -- geometry -------------------------------------------------------------------------------------
    def kp21(self, V):
        """(...,778,3) -> (...,21,3). LINEAR in V, which is why smoothing V smooths the joints identically."""
        J = np.einsum("jv,...vc->...jc", self.J, V)
        return np.concatenate([J, V[..., self.tips, :]], axis=-2)[..., self.kp_map, :]

    def wrist_frames(self, Vw):
        """Wrist pose per frame from the mesh: orientation by weighted Kabsch of the palm against the MANO
        rest pose (weighted by each vertex's skinning weight to the wrist joint, so fingers can't drag it),
        origin at the wrist joint. -> R (N,3,3), t (N,3)."""
        R = np.empty((len(Vw), 3, 3))
        for i, V in enumerate(Vw):
            R[i] = weighted_kabsch(self.v_rest, V, self.W_wrist)
        t = np.einsum("v,...vc->...c", self.J[0], Vw)                   # wrist joint, world
        return R, t

    # -- projective (u, v, Z) parameterisation — the axes the camera actually observed ----------------
    def to_uvz(self, t, focal, cx, cy):
        """metric (X,Y,Z) -> (u px, v px, Z m). u,v are what the detector saw; Z is what it guessed."""
        Z = t[:, 2]
        return np.stack([focal * t[:, 0] / Z + cx, focal * t[:, 1] / Z + cy, Z], 1)

    def from_uvz(self, uvz, focal, cx, cy):
        """(u,v,Z) -> metric (X,Y,Z). Inverse of :meth:`to_uvz`, so a track smoothed in uvz reprojects to
        exactly the smoothed pixel track — the hand cannot drift off the video."""
        Z = uvz[:, 2]
        return np.stack([(uvz[:, 0] - cx) * Z / focal, (uvz[:, 1] - cy) * Z / focal, Z], 1)

    # -- stage 1: jump rejection (in-plane, at the nominal depth — see module docstring) --------------
    def reject_jumps(self, uvz, focal, valid):
        c, out = self.cfg, valid.copy()
        # pixels -> metres of real hand travel, evaluated at the clip's nominal depth. This keeps
        # `jump_threshold` meaning what it means in the reference (metres per frame) without letting the
        # noisy per-frame Z corrupt the gate.
        z_nom = float(np.median(uvz[valid, 2]))
        px_per_m = float(np.median(focal)) / z_nom
        thr_px = c.jump_threshold * px_per_m
        anchor, drops, n_drop, n_re, worst = None, 0, 0, 0, 0.0
        for i in np.flatnonzero(valid):
            if anchor is not None:
                d_px = float(np.linalg.norm(uvz[i, :2] - anchor[:2]))
                d_z = abs(float(uvz[i, 2] - anchor[2]))
                worst = max(worst, d_px)
                if d_px > thr_px or d_z > c.jump_threshold_depth:
                    if drops >= c.jump_reanchor_after:                   # a sustained shift is real motion —
                        anchor, drops, n_re = uvz[i], 0, n_re + 1        # re-anchor, don't cascade-drop
                        continue
                    out[i] = False
                    drops += 1
                    n_drop += 1
                    continue
            anchor, drops = uvz[i], 0
        self.log.append(f"jump-reject: in-plane > {thr_px:.0f} px/frame (= {1e3 * c.jump_threshold:.0f} mm of "
                        f"hand travel at the nominal depth {z_nom:.1f} m) or depth > "
                        f"{1e3 * c.jump_threshold_depth:.0f} mm -> dropped {n_drop}, re-anchored {n_re}; "
                        f"worst in-plane step {worst:.0f} px")
        return out

    # -- stage 2: fill --------------------------------------------------------------------------------
    def fill(self, R, t, L, valid):
        """Fill every internal gap: linear on t and on the wrist-frame articulation L, SLERP on R. Gaps up to
        ``fill_max_gap`` are trustworthy; longer ones are filled anyway (SG needs a continuous signal) but
        flagged so downstream can refuse them. Leading/trailing gaps hold the nearest valid frame."""
        c = self.cfg
        idx = np.flatnonzero(valid)
        if len(idx) < 2:
            raise ValueError("fewer than 2 valid frames — nothing to smooth")
        R, t, L = R.copy(), t.copy(), L.copy()
        trust, short, long_, longest = valid.copy(), 0, 0, 0

        slerp = Slerp(idx, Rotation.from_matrix(R[idx]))
        for a, b in zip(idx[:-1], idx[1:]):
            gap = b - a - 1
            if gap <= 0:
                continue
            g = np.arange(a + 1, b)
            w = ((g - a) / (b - a))[:, None]
            t[g] = (1 - w) * t[a] + w * t[b]
            L[g] = (1 - w)[:, None] * L[a] + w[:, None] * L[b]
            R[g] = slerp(g).as_matrix()
            longest = max(longest, gap)
            if gap <= c.fill_max_gap:
                short += gap
                trust[g] = True                                          # short gap -> trust the interpolation
            else:
                long_ += gap                                             # filled, but NOT trusted
        for g in (np.arange(0, idx[0]), np.arange(idx[-1] + 1, len(valid))):   # hold the nearest valid
            if len(g):
                src = idx[0] if g[0] == 0 else idx[-1]
                R[g], t[g], L[g] = R[src], t[src], L[src]
        self.log.append(f"gap fill: {short} frames in gaps <= {c.fill_max_gap} (trusted), {long_} frames in "
                        f"longer gaps (filled but flagged), longest gap {longest} frames, "
                        f"{len(valid) - idx[-1] - 1 + idx[0]} leading/trailing frames held")
        return R, t, L, trust

    # -- the pipeline ---------------------------------------------------------------------------------
    def run(self, verts, cam_t, found, focal, cx=640.0, cy=360.0):
        c = self.cfg
        verts, cam_t = np.asarray(verts, np.float64), np.asarray(cam_t, np.float64)
        focal = np.asarray(focal, np.float64)
        Vw = verts + cam_t[:, None, :]                                   # camera space
        N = len(Vw)
        if int(np.sum(found)) < c.min_valid_frames:
            raise ValueError(f"only {int(np.sum(found))} valid frames — below min_valid_frames")

        R, t = self.wrist_frames(Vw)
        w0 = t - cam_t                                                   # the root-relative wrist offset
        uvz = self.to_uvz(t, focal, cx, cy)                              # what the camera actually observed
        det = np.asarray(found, bool)
        valid = self.reject_jumps(uvz, focal, det)
        # frames we REPAIRED (detector glitches the gate threw out) vs frames we merely filtered. Keep them
        # apart: any before/after metric that lumps them together is comparing against known-bad data and
        # will read a successful repair as a huge "error".
        rejected = det & ~valid
        self.log.append(f"input: {N} frames, {int(det.sum())} detected, {int(valid.sum())} kept by the gate, "
                        f"{int(rejected.sum())} rejected as detector glitches (repaired by interpolation)")

        # decompose: articulation in the wrist frame (this is the whole point — see the module docstring)
        L = np.einsum("nij,nvj->nvi", np.transpose(R, (0, 2, 1)), Vw - t[:, None, :])
        R, uvz, L, trust = self.fill(R, uvz, L, valid)                   # gaps filled in uvz too — a pixel
        w0 = _interp_invalid(w0, valid)                                  # track interpolates cleanly

        # smooth each part on its own terms, segment by segment (never filter across a hard break)
        uvz_s, R_s, L_s, w0_s = uvz.copy(), R.copy(), L.copy(), w0.copy()
        for a, b in _segments(np.ones(N, bool)):                         # one segment: gaps are filled
            if b - a < c.min_valid_frames:
                continue
            uvz_s[a:b, :2] = _sg(uvz[a:b, :2], c.sg_window, c.sg_polyorder)       # pixels: well conditioned
            uvz_s[a:b, 2] = _sg(uvz[a:b, 2], c.sg_window_depth, c.sg_polyorder)   # depth: noise-dominated
            w0_s[a:b] = _sg(w0[a:b], c.sg_window, c.sg_polyorder)
            R_s[a:b] = (_smooth_rot_sg(R[a:b], c.sg_window, c.sg_polyorder) if c.rot_mode == "sg"
                        else _smooth_rot_ema(R[a:b], c.ema_alpha))
            L_s[a:b] = _sg(L[a:b], c.local_sg_window, c.sg_polyorder)    # <- fingertips, in the wrist frame

        t_s = self.from_uvz(uvz_s, focal, cx, cy)                        # back to metric, projection preserved
        t = self.from_uvz(uvz, focal, cx, cy)                            # (the gap-filled raw, for reporting)
        A = np.einsum("nij,nvj->nvi", R_s, L_s)                          # smoothed mesh, wrist at the origin
        Vw_s = A + t_s[:, None, :]                                       # recompose (camera space)
        # split back into the npz's (root-relative verts, cam_t). J_regressor row 0 sums to 1, so the wrist of
        # `verts` is exactly w0 — carry that through and the two representations stay exactly consistent.
        ct_s = t_s - w0_s
        verts_s = A + w0_s[:, None, :]

        return {
            "verts": verts_s, "cam_t": ct_s, "found": trust,             # <- drop-in for a combined npz
            "wrist_R": R_s, "wrist_t": t_s, "local": L_s,
            "wrist_R_raw": R, "wrist_t_raw": t, "local_raw": L,
            "kp21": self.kp21(Vw_s), "kp21_raw": self.kp21(Vw),
            "tips_local": self.kp21(L_s)[:, [4, 8, 12, 16, 20]],         # fingertips IN THE WRIST FRAME
            "tips_local_raw": self.kp21(L)[:, [4, 8, 12, 16, 20]],
            "verts_world": Vw_s, "verts_world_raw": Vw,
            "valid_in": det, "valid_out": trust, "rejected": rejected,
            "untouched": det & valid,                                    # detected AND kept by the gate
            "focal": focal, "cx": cx, "cy": cy,
            "lin_vel": np.gradient(t_s, self.dt, axis=0),
            "ang_vel": self._ang_vel(R_s),
            "log": list(self.log),
        }

    def _ang_vel(self, R):
        """w = log(R_i^T R_{i+1}) / dt, central where possible (the reference's rotational log-map)."""
        w = np.zeros((len(R), 3))
        rel = Rotation.from_matrix(np.transpose(R[:-1], (0, 2, 1)) @ R[1:]).as_rotvec() / self.dt
        w[1:-1] = 0.5 * (rel[:-1] + rel[1:])
        if len(R) > 1:
            w[0], w[-1] = rel[0], rel[-1]
        return w


# ----------------------------------------------------------------------------------------------------
# HaWoR world-frame Butterworth filtering
# ----------------------------------------------------------------------------------------------------
@dataclass
class ButterworthCfg:
    """Offline filter used by cop-viz after a tracker has produced its raw archive."""

    cutoff_hz: float = 2.0
    prototype_order: int = 2
    max_interpolated_gap: int = 3
    minimum_run_frames: int | None = None


def _bw_palm_frames(joints):
    """Palm-defined wrist-to-world frames from wrist and index/middle/pinky MCPs."""
    joints = np.asarray(joints, np.float64)
    if joints.ndim != 3 or joints.shape[1] <= 17 or joints.shape[2] != 3:
        raise ValueError("joints must have shape (T, J>=18, 3)")
    frames = np.full((len(joints), 3, 3), np.nan, np.float64)
    valid = np.zeros(len(joints), bool)
    for i in range(len(joints)):
        wrist, index_mcp, middle_mcp, pinky_mcp = joints[i, [0, 5, 9, 17]]
        if not np.all(np.isfinite((wrist, index_mcp, middle_mcp, pinky_mcp))):
            continue
        x = index_mcp - pinky_mcp
        nx = np.linalg.norm(x)
        if nx <= 1e-8:
            continue
        x /= nx
        y = middle_mcp - wrist
        y -= np.dot(y, x) * x
        ny = np.linalg.norm(y)
        if ny <= 1e-8:
            continue
        y /= ny
        z = np.cross(x, y)
        nz = np.linalg.norm(z)
        if nz <= 1e-8:
            continue
        z /= nz
        y = np.cross(z, x)
        frames[i] = np.column_stack((x, y, z))
        valid[i] = True
    return frames, valid


def _bw_to_wrist(points, wrist, frames):
    return np.einsum("tji,tnj->tni", frames, points - wrist[:, None, :])


def _bw_from_wrist(points, wrist, frames):
    return np.einsum("tij,tnj->tni", frames, points) + wrist[:, None, :]


def _bw_runs(mask):
    mask = np.asarray(mask, bool).reshape(-1)
    edges = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _bw_sos_padlen(sos):
    zeros = int(np.count_nonzero(sos[:, 2] == 0))
    poles = int(np.count_nonzero(sos[:, 5] == 0))
    return 3 * (2 * len(sos) + 1 - min(zeros, poles))


def _bw_trusted_interpolated(observed, interpolated, max_gap):
    observed = np.asarray(observed, bool).reshape(-1)
    interpolated = np.asarray(interpolated, bool).reshape(-1)
    if observed.shape != interpolated.shape:
        raise ValueError("observed and interpolated masks differ in shape")
    if np.any(observed & interpolated):
        raise ValueError("observed and interpolated masks overlap")
    trusted = np.zeros_like(observed)
    for start, end in _bw_runs(interpolated):
        if start > 0 and end < len(observed) and observed[start - 1] and observed[end] \
                and end - start <= max_gap:
            trusted[start:end] = True
    return trusted


def _bw_filter_runs(values, support, sos, minimum_run_frames):
    values = np.asarray(values, np.float64)
    support = np.asarray(support, bool).reshape(-1)
    finite = np.all(np.isfinite(values), axis=tuple(range(1, values.ndim)))
    support = support & finite
    output = values.copy()
    applied = np.zeros(len(values), bool)
    edge_sensitive = np.zeros(len(values), bool)
    padlen = _bw_sos_padlen(sos)
    if minimum_run_frames <= padlen:
        raise ValueError(f"minimum_run_frames must exceed filter pad length {padlen}")
    filtered_runs, skipped_runs = [], []
    for start, end in _bw_runs(support):
        record = {"start": int(start), "end": int(end - 1), "frames": int(end - start)}
        if end - start < minimum_run_frames:
            skipped_runs.append(record)
            continue
        output[start:end] = sosfiltfilt(sos, values[start:end], axis=0, padlen=padlen)
        applied[start:end] = True
        edge_sensitive[start:start + padlen] = True
        edge_sensitive[end - padlen:end] = True
        filtered_runs.append(record)
    return output, applied, edge_sensitive, {
        "padlen": padlen,
        "minimum_run_frames": minimum_run_frames,
        "support_frames": int(support.sum()),
        "filtered_runs": filtered_runs,
        "skipped_runs": skipped_runs,
    }


def _bw_summary(values):
    values = np.asarray(values, np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"samples": 0, "mean": None, "median": None, "p95": None, "max": None, "rms": None}
    return {
        "samples": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        "rms": float(np.sqrt(np.mean(values ** 2))),
    }


class ButterworthHandFilter:
    """Filter raw HaWoR geometry while keeping tracker output and filter ownership separate.

    Wrist translation is filtered in HaWoR's metric 3D world frame. Hand articulation is
    filtered in the palm-defined wrist frame, then recomposed and transformed back to the
    source camera. Filtering all vertices makes the mesh and CoPs consistent with the five
    fingertip tracks; the reported fingertip benchmark uses joints 4/8/12/16/20.
    """

    TIP_INDICES = np.array([4, 8, 12, 16, 20], np.int64)
    FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

    def __init__(self, fps=24.0, cfg: ButterworthCfg | None = None):
        self.fps = float(fps)
        self.cfg = cfg or ButterworthCfg()
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        if not 0 < self.cfg.cutoff_hz < self.fps / 2:
            raise ValueError("cutoff_hz must be between zero and Nyquist")
        if self.cfg.prototype_order < 1:
            raise ValueError("prototype_order must be positive")
        correction = (np.sqrt(2.0) - 1.0) ** (1.0 / (2.0 * self.cfg.prototype_order))
        self.design_cutoff_hz = self.fps / np.pi * np.arctan(
            np.tan(np.pi * self.cfg.cutoff_hz / self.fps) / correction
        )
        if self.design_cutoff_hz >= self.fps / 2:
            raise ValueError("compensated Butterworth cutoff reaches Nyquist")
        self.sos = butter(
            self.cfg.prototype_order,
            self.design_cutoff_hz,
            btype="lowpass",
            fs=self.fps,
            output="sos",
        )
        self.minimum_run_frames = self.cfg.minimum_run_frames or max(
            _bw_sos_padlen(self.sos) + 1, int(np.ceil(self.fps)) + 1
        )

    @staticmethod
    def _camera_to_world(points, rotation, translation):
        return np.einsum("tij,tnj->tni", rotation, points) + translation[:, None, :]

    @staticmethod
    def _world_to_camera(points, rotation, translation):
        return np.einsum("tij,tnj->tni", rotation, points) + translation[:, None, :]

    def _velocity_benchmark(self, raw_wrist, filtered_wrist, raw_tips, filtered_tips, mask):
        windows = np.asarray(mask[1:] & mask[:-1], bool)
        raw_wrist_speed = np.linalg.norm(np.diff(raw_wrist, axis=0), axis=-1) * self.fps
        filtered_wrist_speed = np.linalg.norm(np.diff(filtered_wrist, axis=0), axis=-1) * self.fps
        raw_tip_speed = np.linalg.norm(np.diff(raw_tips, axis=0), axis=-1) * self.fps
        filtered_tip_speed = np.linalg.norm(np.diff(filtered_tips, axis=0), axis=-1) * self.fps

        def comparison(raw, filtered):
            raw_summary, filtered_summary = _bw_summary(raw), _bw_summary(filtered)
            raw_mean, filtered_mean = raw_summary["mean"], filtered_summary["mean"]
            reduction = None if raw_mean in (None, 0.0) else 100.0 * (1.0 - filtered_mean / raw_mean)
            return {"raw": raw_summary, "filtered": filtered_summary, "mean_reduction_percent": reduction}

        fingertips = comparison(raw_tip_speed[windows], filtered_tip_speed[windows])
        fingertips["per_finger"] = {
            name: comparison(raw_tip_speed[windows, i], filtered_tip_speed[windows, i])
            for i, name in enumerate(self.FINGER_NAMES)
        }
        return {
            "units": "m/s",
            "evaluated_frame_pairs": int(windows.sum()),
            "wrist_world": comparison(raw_wrist_speed[windows], filtered_wrist_speed[windows]),
            "fingertips_wrist": fingertips,
        }

    def run(self, verts_cam, joints_cam, observed, interpolated,
            cam_R_c2w, cam_t_c2w, cam_R_w2c, cam_t_w2c):
        verts_cam = np.asarray(verts_cam, np.float64)
        joints_cam = np.asarray(joints_cam, np.float64)
        n = len(verts_cam)
        if verts_cam.ndim != 3 or verts_cam.shape[1:] != (778, 3):
            raise ValueError(f"verts_cam must have shape (T, 778, 3), got {verts_cam.shape}")
        if joints_cam.ndim != 3 or joints_cam.shape[0] != n or joints_cam.shape[2] != 3:
            raise ValueError(f"joints_cam must have shape (T, J, 3), got {joints_cam.shape}")
        transforms = [
            np.asarray(cam_R_c2w, np.float64), np.asarray(cam_t_c2w, np.float64),
            np.asarray(cam_R_w2c, np.float64), np.asarray(cam_t_w2c, np.float64),
        ]
        if transforms[0].shape != (n, 3, 3) or transforms[1].shape != (n, 3) \
                or transforms[2].shape != (n, 3, 3) or transforms[3].shape != (n, 3):
            raise ValueError("camera transforms must be per-frame rigid transforms")
        R_c2w, t_c2w, R_w2c, t_w2c = transforms
        inverse_error = float(np.max(np.abs(np.einsum("tij,tjk->tik", R_w2c, R_c2w) - np.eye(3))))
        translation_error = float(np.max(np.abs(t_w2c + np.einsum("tij,tj->ti", R_w2c, t_c2w))))
        if max(inverse_error, translation_error) > 1e-5:
            raise ValueError("camera world transforms are not mutual inverses")

        joints_world = self._camera_to_world(joints_cam, R_c2w, t_c2w)
        verts_world = self._camera_to_world(verts_cam, R_c2w, t_c2w)
        roundtrip = self._world_to_camera(joints_world, R_w2c, t_w2c)
        roundtrip_error = float(np.max(np.abs(roundtrip - joints_cam)))
        if roundtrip_error > 1e-5:
            raise ValueError("camera/world round trip exceeds tolerance")

        frames, frame_valid = _bw_palm_frames(joints_world)
        observed = np.asarray(observed, bool).reshape(-1)
        interpolated = np.asarray(interpolated, bool).reshape(-1)
        if observed.shape != (n,) or interpolated.shape != (n,):
            raise ValueError("observation masks must have one value per frame")
        trusted = _bw_trusted_interpolated(
            observed, interpolated, self.cfg.max_interpolated_gap
        )
        geometry_valid = np.all(np.isfinite(joints_world), axis=(1, 2))
        geometry_valid &= np.all(np.isfinite(verts_world), axis=(1, 2))
        support = (observed | trusted) & frame_valid & geometry_valid

        raw_wrist = joints_world[:, 0]
        raw_joints_wrist = _bw_to_wrist(joints_world, raw_wrist, frames)
        raw_verts_wrist = _bw_to_wrist(verts_world, raw_wrist, frames)
        filtered_wrist, wrist_applied, edge_sensitive, run_policy = _bw_filter_runs(
            raw_wrist, support, self.sos, self.minimum_run_frames
        )
        filtered_joints_wrist, joint_applied, joint_edges, _ = _bw_filter_runs(
            raw_joints_wrist, support, self.sos, self.minimum_run_frames
        )
        filtered_verts_wrist, vertex_applied, vertex_edges, _ = _bw_filter_runs(
            raw_verts_wrist, support, self.sos, self.minimum_run_frames
        )
        if not (np.array_equal(wrist_applied, joint_applied)
                and np.array_equal(wrist_applied, vertex_applied)
                and np.array_equal(edge_sensitive, joint_edges)
                and np.array_equal(edge_sensitive, vertex_edges)):
            raise ValueError("filter support diverged across wrist, joints, and vertices")

        filtered_joints_world = joints_world.copy()
        filtered_verts_world = verts_world.copy()
        reconstructable = frame_valid & geometry_valid
        joints_reconstructed = _bw_from_wrist(
            filtered_joints_wrist, filtered_wrist, frames
        )
        verts_reconstructed = _bw_from_wrist(
            filtered_verts_wrist, filtered_wrist, frames
        )
        filtered_joints_world[reconstructable] = joints_reconstructed[reconstructable]
        filtered_verts_world[reconstructable] = verts_reconstructed[reconstructable]
        filtered_joints_cam = self._world_to_camera(
            filtered_joints_world, R_w2c, t_w2c
        )
        filtered_verts_cam = self._world_to_camera(
            filtered_verts_world, R_w2c, t_w2c
        )

        metric_mask = wrist_applied & ~edge_sensitive
        tips = self.TIP_INDICES
        benchmark = self._velocity_benchmark(
            raw_wrist,
            filtered_wrist,
            raw_joints_wrist[:, tips],
            filtered_joints_wrist[:, tips],
            metric_mask,
        )
        return {
            "verts_cam": filtered_verts_cam,
            "joints3d_cam": filtered_joints_cam,
            "verts_world": filtered_verts_world,
            "joints3d_world": filtered_joints_world,
            "raw_wrist_world": raw_wrist,
            "filtered_wrist_world": filtered_wrist,
            "raw_fingertips_wrist": raw_joints_wrist[:, tips],
            "filtered_fingertips_wrist": filtered_joints_wrist[:, tips],
            "trusted_interpolated": trusted,
            "support": support,
            "applied": wrist_applied,
            "edge_sensitive": edge_sensitive,
            "metric_mask": metric_mask,
            "velocity_benchmark": benchmark,
            "filter": {
                "type": "Butterworth low-pass",
                "cutoff_hz": self.cfg.cutoff_hz,
                "design_cutoff_hz": float(self.design_cutoff_hz),
                "prototype_order": self.cfg.prototype_order,
                "effective_zero_phase_order": 2 * self.cfg.prototype_order,
                "zero_phase": True,
                "fps": self.fps,
                "sos": np.asarray(self.sos).tolist(),
                "wrist_frame": (
                    "x=index_MCP-pinky_MCP, y=orthogonalized wrist-to-middle_MCP, z=x cross y"
                ),
                "run_policy": run_policy,
            },
            "validation": {
                "camera_rotation_inverse_max_abs": inverse_error,
                "camera_translation_inverse_max_abs": translation_error,
                "camera_world_roundtrip_max_abs": roundtrip_error,
                "palm_frame_valid_frames": int(frame_valid.sum()),
            },
        }
