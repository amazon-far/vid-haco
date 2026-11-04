"""``HandSource`` adapter: a HACO_seq **combined npz** -> the cop-viz fit/viz stack.

The HACO_seq deliverable (``<TAG>_combined.npz``, one row per video frame) carries the WiLoR MANO hand
(``verts``/``cam_t``/``focal``/``found``) and the HACO per-vertex contact map (``contact_prob``), but no
skeleton, no part labels and no per-finger contact points — so it does not yet speak cop-viz's protocol.
This adapter closes that gap with nothing but numpy/scipy + the baked ``mano_assets.npz``:

  kp21(i)                       J_regressor @ verts (16 MANO joints) + the 5 tip vertices, reordered to
                                the kp21 layout (0=wrist, then MCP/PIP/DIP/TIP per finger)
  targets(i)                    the contact map, SEGMENTED BY FINGER (per-vertex LBS part labels) ->
                                one contact per finger: CoP = the prob-weighted centroid of that finger's
                                contacting vertices, ``normal`` = its prob-weighted surface normal
  snap_points_to_source(i,pts)  exact closest point on the MANO surface (+ that triangle's normal)
  nearest_vertex / vertex_world the vertex-lock hooks — so the fitter can glue a CoP to ONE material vertex

Frames where WiLoR lost the hand (``found=False``) report ``kp21 -> None`` / no active contact, which the
fitter is documented to tolerate.

CONTACT MASK — reproduces the deliverable's OWN post-processing (``lib/utils/contact_mask.py``, as used by
``render_meshtop.py`` for the rendered videos): causal EMA (alpha=0.8) over ``contact_prob >= thr``,
then drop connected components smaller than 20 vertices over the WATERTIGHT face graph. So the contact this
harness fits CoPs to is exactly the contact the released video shows.

SIGN CONVENTION (cop-viz's, see ``closure_force``): ``normal`` is the outward normal OF THE OBJECT the hand
presses on — i.e. the hand-surface normal FLIPPED to point into the hand — so that the push
``force_dir = -normal`` points OUT of the finger pad, into the object. ``FixedCopFitter.apply`` sign-locks
to the same rule (``normal . force_dir < 0``), so it is stable under the fit.

FRAME — the npz stores the hand ROOT-RELATIVE (metres, +Y down) plus a per-frame ``cam_t`` that places it in
front of the camera. We fit and visualize in the ROOT-RELATIVE frame, and use ``cam_t`` only to project back
onto the video (``project()``), because ``cam_t`` is NOT a usable metric world frame: it is WiLoR's weak-
perspective depth (z ~ 16.6 +- 1.9 m at focal 25000 px), so it jumps a median of 100 mm PER FRAME — 50x the
hand's actual 2 mm/frame articulation. Adding it would drown every CoP in camera noise. Pass
``camera_space=True`` to fit in camera space anyway (and watch the jitter numbers explode).

This module is also the command-line combiner for HaWoR-backed archives. It validates and concatenates
ordered HACO contact shards, maps HaWoR's local rows onto an explicit absolute source-frame interval, and
emits the same combined schema consumed by ``HacoSeqSource``. HaWoR's metric world/camera arrays and optional
Orient Anything ``object_*`` metadata ride alongside that compatibility schema without being discarded.
"""
from __future__ import annotations

import argparse
import os
import os.path as op

import numpy as np

HERE = op.dirname(op.abspath(__file__))
DEFAULT_ASSETS = op.join(HERE, "..", "data", "base_data", "mano_assets.npz")   # tools/make_mano_assets.py
DEFAULT_NPZ = "out/example/example_vitl_combined.npz"
MANO_VERTICES = 778
MANO_JOINTS = 21

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]          # == cop_viz.skeleton.FINGERS
PARTS = ["palm"] + FINGERS                                       # part ids 0..5 in mano_assets.vert_finger


def _unit(v, eps=1e-12):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + eps)


def coded_frames(path, upto=None):
    """Iterate a source video's CODED frames — exactly one per row of the npz. Yields (index, rgb uint8).

    NOT imageio's default reader. These clips are VARIABLE-frame-rate (IMG_3970: 22.53 fps, tbr 120), and
    ffmpeg's default constant-frame-rate conversion DUPLICATES frames to fill the higher rate — it hands you
    40338 frames instead of 7575, so every frame index silently points at the wrong moment (the overlay lands
    on a different part of the video and nothing errors). ``-vsync 0`` passes the coded frames through, which
    is what cv2 — and therefore Stage A and the deliverable's own renderer — saw. Callers should check they
    got every frame they asked for."""
    import imageio_ffmpeg

    rd = imageio_ffmpeg.read_frames(path, output_params=["-vsync", "0"], bits_per_pixel=24)
    W, H = next(rd)["size"]
    for i, buf in enumerate(rd):
        if upto is not None and i > upto:
            break
        yield i, np.frombuffer(buf, np.uint8).reshape(H, W, 3)


def vertex_normals(V, F):
    """Area-weighted per-vertex normals of a mesh (V (n,3), F (f,3)) -> (n,3) unit, outward (MANO's faces
    are consistently wound, so the accumulated normals point out of the hand)."""
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])   # |fn| = 2 * area  -> area weighting
    N = np.zeros_like(V)
    for c in range(3):
        np.add.at(N, F[:, c], fn)
    return _unit(N)


def closest_points_on_mesh(P, V, F):
    """Exact closest point on a triangle mesh for each of ``P`` (m,3) -> (points (m,3), face_ids (m,)).

    Brute force over all faces (MANO has 1538, and a frame has at most 6 CoPs — one broadcast numpy pass
    beats any acceleration structure and keeps this numpy-only). Ericson, *Real-Time Collision Detection*
    5.1.5, vectorized over BOTH points and faces: a triangle's closest point lies in one of 7 Voronoi
    regions (3 vertices, 3 edges, the interior)."""
    P = np.atleast_2d(np.asarray(P, float))[:, None, :]           # (m,1,3)
    A, B, C = (V[F[:, i]][None] for i in range(3))                # (1,f,3) each
    ab, ac = B - A, C - A
    ap, bp, cp = P - A, P - B, P - C
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)                 # (m,f)
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2

    def _div(n, d):
        return np.divide(n, d, out=np.zeros_like(n), where=np.abs(d) > 1e-20)

    v, w = _div(vb, va + vb + vc), _div(vc, va + vb + vc)
    Q = A + v[..., None] * ab + w[..., None] * ac                 # default region: the face interior
    for cond, pt in (                                             # then the edges, then the vertices (last
        ((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0),             # match wins, matching Ericson's order)
         B + _div(d4 - d3, (d4 - d3) + (d5 - d6))[..., None] * (C - B)),               # edge BC
        ((vb <= 0) & (d2 >= 0) & (d6 <= 0), A + _div(d2, d2 - d6)[..., None] * ac),    # edge AC
        ((vc <= 0) & (d1 >= 0) & (d3 <= 0), A + _div(d1, d1 - d3)[..., None] * ab),    # edge AB
        ((d6 >= 0) & (d5 <= d6), np.broadcast_to(C, Q.shape)),                         # vertex C
        ((d3 >= 0) & (d4 <= d3), np.broadcast_to(B, Q.shape)),                         # vertex B
        ((d1 <= 0) & (d2 <= 0), np.broadcast_to(A, Q.shape)),                          # vertex A
    ):
        Q = np.where(cond[..., None], pt, Q)
    fid = np.argmin(((Q - P) ** 2).sum(-1), axis=1)               # (m,) winning face per point
    return Q[np.arange(len(fid)), fid], fid


class HacoSeqSource:
    """The HACO_seq combined npz as a cop-viz ``HandSource``.

    ``min_verts``  a finger counts as in contact once this many of its vertices are in the (post-processed)
                   contact mask — filters the odd single-vertex speckle that survives component filtering.
    """

    def __init__(self, npz_path=DEFAULT_NPZ, assets_path=DEFAULT_ASSETS, *, thr=None, min_verts=3,
                 ema_alpha=0.8, min_component=20, camera_space=False):
        z = np.load(npz_path, allow_pickle=True)
        a = np.load(assets_path, allow_pickle=True)
        self.path = npz_path
        self.faces = np.asarray(z["faces"], np.int64)
        self.hand_side = str(z["hand_side"]) if "hand_side" in z.files else "right"
        self.W, self.H = (int(z["img_wh"][0]), int(z["img_wh"][1])) if "img_wh" in z.files else (1280, 720)
        self.fps = float(z["fps"])
        self.thr = float(z["thr"]) if thr is None else float(thr)
        self.found = np.asarray(z["found"], bool)
        self.cam_t = np.asarray(z["cam_t"], np.float64)
        self.focal = np.asarray(z["focal"], np.float64)
        self.bbox_sm = np.asarray(z["bbox_sm"], np.float64)          # (x,y,w,h) in full-res video pixels
        self.bbox_xyxy = np.asarray(z["bbox_xyxy"], np.float64)      # WiLoR's raw detection box
        # ABSOLUTE video frame per row. For the full deliverable this is just 0..N-1, but a CROPPED clip
        # (make_clips.py) keeps the original ids — so row k is video frame frame_idx[k], not frame k.
        # Everything else in this class is indexed POSITIONALLY; only the video needs this map.
        self.frame_idx = np.asarray(z["frame_idx"], np.int64) if "frame_idx" in z.files \
            else np.arange(len(z["verts"]), dtype=np.int64)
        self.contact_prob = np.asarray(z["contact_prob"], np.float64)
        self.hand_contact = np.asarray(z["hand_contact"], np.float64)
        # ROOT-RELATIVE by default (see the module docstring: cam_t is too noisy to fit in). The
        # deliverable's own renderer meshes `verts + cam_t` — that is exactly what `project()` does.
        self.verts = np.asarray(z["verts"], np.float64) + (self.cam_t[:, None, :] if camera_space else 0.0)
        self.camera_space = bool(camera_space)
        self.N = len(self.verts)
        self.min_verts = int(min_verts)

        self.vert_finger = np.asarray(a["vert_finger"], np.int64)          # 0 palm, 1..5 thumb..pinky
        self.part_sel = [self.vert_finger == i for i in range(6)]
        wt = np.asarray(a["watertight_faces"], np.int64)
        if self.hand_side == "left":                # left hand: npz verts are x-mirrored + faces reversed, so
            wt = wt[:, ::-1]                         # orient the asset's watertight faces to match (winding-
        self.watertight_faces = wt                  # sensitive uses: run_cop_fit STAGE-0 signed volume)

        # kp21 for every frame at once: 16 MANO joints (J_regressor @ verts) + the 5 tip vertices,
        # reordered to the kp21 layout the cop-viz skeleton assumes.
        J = np.einsum("jv,nvc->njc", np.asarray(a["J_regressor"], np.float64), self.verts)   # (N,16,3)
        tips = self.verts[:, np.asarray(a["tip_vids"], np.int64)]                            # (N,5,3)
        self.kp21_all = np.concatenate([J, tips], 1)[:, np.asarray(a["kp21_map"], np.int64)]  # (N,21,3)

        self.cmask = self._contact_mask(ema_alpha, min_component, wt)
        self._vn_cache, self._kd_cache = {}, {}

    # ---- contact mask (the deliverable's own post-processing) ------------------------------------
    def _contact_mask(self, alpha, min_component, wt_faces):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        raw = self.contact_prob >= self.thr                                # (N,778)
        sm = None
        out = np.zeros_like(raw)
        for i in range(self.N):                                            # causal EMA, exactly as released
            sm = raw[i].astype(np.float32) if sm is None else alpha * sm + (1 - alpha) * raw[i]
            out[i] = sm > 0.5
        # face-adjacency graph (watertight: the wrist hole sealed, as the released renderer used)
        e = np.concatenate([wt_faces[:, [0, 1]], wt_faces[:, [1, 2]], wt_faces[:, [2, 0]]])
        n = self.contact_prob.shape[1]
        A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
        A = A + A.T
        for i in range(self.N):                                            # drop components < min_component
            idx = np.flatnonzero(out[i])
            if len(idx) == 0:
                continue
            _, lab = connected_components(A[idx][:, idx], directed=False)
            keep = np.bincount(lab)[lab] >= min_component
            out[i] = False
            out[i, idx[keep]] = True
        return out

    def project(self, pts=None, W=None, H=None, idx=None):
        """Project into the source video the SAME way the deliverable's renderer does: camera space
        (``verts + cam_t``), pinhole fx=fy=``focal``, principal point at the image centre. ``pts`` defaults
        to the hand verts; ``idx`` selects frames (default: all). -> (u, v) each (frames, n_points)."""
        W, H = (self.W if W is None else W), (self.H if H is None else H)
        idx = np.arange(self.N) if idx is None else np.atleast_1d(idx)
        P = (self.verts[idx] if pts is None else np.asarray(pts, float)[idx])
        if not self.camera_space:
            P = P + self.cam_t[idx][:, None, :]
        f = self.focal[idx][:, None]
        return f * P[..., 0] / P[..., 2] + W / 2.0, f * P[..., 1] / P[..., 2] + H / 2.0

    def project_points(self, i, pts, W=None, H=None):
        """Project arbitrary points of frame ``i`` (a CoP, a force-arrow tip, ...) -> (n,2) pixels."""
        W, H = (self.W if W is None else W), (self.H if H is None else H)
        P = np.atleast_2d(np.asarray(pts, float))
        if not self.camera_space:
            P = P + self.cam_t[i]
        f = self.focal[i]
        return np.stack([f * P[:, 0] / P[:, 2] + W / 2.0, f * P[:, 1] / P[:, 2] + H / 2.0], 1)

    def crop_boxes(self, frames, *, pad=1.6, ema=0.85, W=None, H=None):
        """Per-frame SQUARE crop windows around the hand -> (T,3) int (cx, cy, half_side).

        WiLoR's per-frame box jitters hard; an unsmoothed crop is unwatchable. EMA the centre and the size,
        square it, pad it, and clamp it inside the frame."""
        W, H = (self.W if W is None else W), (self.H if H is None else H)
        b = self.bbox_xyxy[np.asarray(frames)]
        out = np.zeros((len(b), 3), int)
        cx = cy = half = None
        for k, (x0, y0, x1, y1) in enumerate(b):
            c0, c1, h = (x0 + x1) / 2, (y0 + y1) / 2, pad * max(x1 - x0, y1 - y0) / 2
            if cx is None:
                cx, cy, half = c0, c1, h
            else:
                cx, cy, half = (ema * cx + (1 - ema) * c0, ema * cy + (1 - ema) * c1, ema * half + (1 - ema) * h)
            hi = int(max(32, min(half, min(W, H) / 2)))
            out[k] = (int(np.clip(cx, hi, W - hi)), int(np.clip(cy, hi, H - hi)), hi)
        return out

    # ---- cached per-frame geometry ---------------------------------------------------------------
    def normals(self, i):
        if i not in self._vn_cache:
            if len(self._vn_cache) > 64:
                self._vn_cache.clear()
            self._vn_cache[i] = vertex_normals(self.verts[i], self.faces)
        return self._vn_cache[i]

    def _kd(self, i):
        from scipy.spatial import cKDTree

        if i not in self._kd_cache:
            if len(self._kd_cache) > 64:
                self._kd_cache.clear()
            self._kd_cache[i] = cKDTree(self.verts[i])
        return self._kd_cache[i]

    # ---- HandSource protocol ---------------------------------------------------------------------
    def kp21(self, i):
        """(21,3) skeleton in metres, or None where WiLoR lost the hand (the fitter tolerates None)."""
        return self.kp21_all[i] if self.found[i] else None

    def targets(self, i):
        """{finger: {active, cop, normal, force_dir, force_mag, ...}} from the contact map.

        One contact per finger (+ ``palm``, which rides along for the viz but is not fitted): the CoP is
        the contact-probability-weighted centroid of that finger's contacting vertices, its ``normal`` the
        weighted surface normal (flipped to the object-outward convention), and ``force_mag`` the
        probability mass in contact (~ area x confidence). ``force_dir`` is the NAIVE pre-closure push —
        "toward the grasp centre" — which ``closure_force`` then replaces with a friction-cone fit; the
        two are compared in ``run_cop_fit.py``.
        """
        out = {p: {"active": False} for p in PARTS}
        if not self.found[i]:
            return out
        m, prob, VN = self.cmask[i], self.contact_prob[i], None
        for pid, name in enumerate(PARTS):
            sel = np.flatnonzero(m & self.part_sel[pid])
            if len(sel) < self.min_verts:
                continue
            if VN is None:
                VN = self.normals(i)
            w = prob[sel]
            cop = (self.verts[i][sel] * w[:, None]).sum(0) / w.sum()      # inside the pad; snapped by the fit
            n_hand = _unit((VN[sel] * w[:, None]).sum(0))                 # out of the HAND
            out[name] = {"active": True, "cop": cop, "normal": -n_hand,   # cop-viz sign: push = -normal
                         "force_dir": -(-n_hand), "force_mag": float(w.sum()),
                         "n_verts": int(len(sel)), "vids": sel}
        act = [f for f in FINGERS if out[f]["active"]]
        if len(act) >= 2:                                                  # NAIVE push: toward the grasp centre
            ctr = np.mean([out[f]["cop"] for f in act], 0)
            for f in act:
                d = ctr - out[f]["cop"]
                if np.linalg.norm(d) > 1e-6:
                    out[f]["force_dir"] = _unit(d)
        return out

    def snap_points_to_source(self, i, pts):
        """Project points onto the MANO surface -> (on_surface (n,3), normals (n,3)). Normals are the
        triangle normals, flipped to the object-outward convention (into the hand), matching ``targets``."""
        V, F = self.verts[i], self.faces
        on, fid = closest_points_on_mesh(pts, V, F)
        fn = np.cross(V[F[fid, 1]] - V[F[fid, 0]], V[F[fid, 2]] - V[F[fid, 0]])
        return on, -_unit(fn)

    def nearest_vertex(self, i, pts):
        """(n,) MANO vertex ids nearest to ``pts`` — the vertex-lock hook."""
        return self._kd(i).query(np.atleast_2d(np.asarray(pts, float)))[1]

    def vertex_world(self, i, vids):
        """(verts (n,3), normals (n,3)) of the given MANO vertices this frame — the vertex-lock hook."""
        vids = np.asarray(vids, int)
        return self.verts[i][vids], -self.normals(i)[vids]                # object-outward, as above


# ---- HaWoR + HACO combined-archive CLI ----------------------------------------------------------
def _load_archive(path, label, keys=None):
    path = op.abspath(op.expanduser(str(path)))
    if not op.isfile(path):
        raise ValueError(f"{label} does not exist: {path}")
    try:
        with np.load(path, allow_pickle=True) as z:
            selected = z.files if keys is None else [key for key in keys if key in z.files]
            return path, {key: np.asarray(z[key]) for key in selected}
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read {label} {path}: {exc}") from exc


def _require_keys(data, keys, label):
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"{label} is missing keys: {', '.join(missing)}")


def _scalar(data, key, label):
    _require_keys(data, [key], label)
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"{label}:{key} must be scalar, got shape {value.shape}")
    return value.reshape(-1)[0].item()


def _scalars_equal(a, b):
    if isinstance(a, (bool, int, float, np.number)) and isinstance(b, (bool, int, float, np.number)):
        return bool(np.isclose(a, b, rtol=0.0, atol=1e-6, equal_nan=False))
    return a == b


def _contiguous_frame_idx(value, label):
    frame_idx = np.asarray(value)
    if frame_idx.ndim != 1 or not np.issubdtype(frame_idx.dtype, np.integer):
        raise ValueError(f"{label}:frame_idx must be a 1D integer array, got {frame_idx.shape} {frame_idx.dtype}")
    if len(frame_idx) == 0:
        raise ValueError(f"{label}:frame_idx is empty")
    frame_idx = frame_idx.astype(np.int64, copy=False)
    gaps = np.flatnonzero(np.diff(frame_idx) != 1)
    if len(gaps):
        i = int(gaps[0])
        raise ValueError(
            f"{label}:frame_idx is not strictly contiguous at rows {i}/{i + 1}: "
            f"{int(frame_idx[i])}, {int(frame_idx[i + 1])}"
        )
    return frame_idx


def _check_probability_array(value, shape, label):
    value = np.asarray(value)
    if value.shape != shape:
        raise ValueError(f"{label} must have shape {shape}, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{label} contains non-finite values")
    if np.any(value < -1e-6) or np.any(value > 1.0 + 1e-6):
        lo, hi = float(value.min()), float(value.max())
        raise ValueError(f"{label} is outside probability range [0, 1]: min={lo:g}, max={hi:g}")
    return value


def _load_contact_shards(paths):
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    if not paths:
        raise ValueError("at least one ordered HACO contact shard is required")
    frames, contact_prob, hand_contact = [], [], []
    first_scalars = None
    resolved = []
    previous_last = None
    for shard_number, path in enumerate(paths):
        path, data = _load_archive(path, f"contact shard {shard_number}")
        label = f"contact shard {shard_number} ({path})"
        _require_keys(data, ["frame_idx", "contact_prob", "hand_contact", "thr", "clip_len", "stride"], label)
        frame_idx = _contiguous_frame_idx(data["frame_idx"], label)
        if previous_last is not None and int(frame_idx[0]) != previous_last + 1:
            raise ValueError(
                f"contact shards are not in contiguous order: previous ends at {previous_last}, "
                f"{op.basename(path)} starts at {int(frame_idx[0])}"
            )
        previous_last = int(frame_idx[-1])
        n = len(frame_idx)
        cp = _check_probability_array(data["contact_prob"], (n, MANO_VERTICES), f"{label}:contact_prob")
        hc = _check_probability_array(data["hand_contact"], (n, 2), f"{label}:hand_contact")
        scalars = {key: _scalar(data, key, label) for key in ("thr", "clip_len", "stride")}
        if not 0.0 <= float(scalars["thr"]) <= 1.0:
            raise ValueError(f"{label}:thr must be in [0, 1], got {scalars['thr']}")
        if int(scalars["clip_len"]) <= 0 or int(scalars["stride"]) <= 0:
            raise ValueError(
                f"{label}:clip_len and stride must be positive, got "
                f"{scalars['clip_len']} and {scalars['stride']}"
            )
        if first_scalars is None:
            first_scalars = scalars
        else:
            for key, value in scalars.items():
                if not _scalars_equal(value, first_scalars[key]):
                    raise ValueError(
                        f"{label}:{key}={value!r} differs from first shard value {first_scalars[key]!r}"
                    )
        frames.append(frame_idx)
        contact_prob.append(cp.astype(np.float32, copy=False))
        hand_contact.append(hc.astype(np.float32, copy=False))
        resolved.append(path)

    return {
        "frame_idx": np.concatenate(frames),
        "contact_prob": np.concatenate(contact_prob),
        "hand_contact": np.concatenate(hand_contact),
        "thr": np.asarray(first_scalars["thr"], dtype=np.float32),
        "clip_len": np.asarray(first_scalars["clip_len"], dtype=np.int64),
        "stride": np.asarray(first_scalars["stride"], dtype=np.int64),
        "haco_contact_shards": np.asarray(resolved),
    }


def _load_haco_meta(paths, expected_frames):
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    if not paths:
        return {}
    frames, boxes, resolved = [], [], []
    first_fps, first_img_wh = None, None
    previous_last = None
    for shard_number, path in enumerate(paths):
        path, data = _load_archive(
            path,
            f"HACO metadata archive {shard_number}",
            keys=("frame_idx", "bbox_sm", "fps", "img_wh"),
        )
        label = f"HACO metadata archive {shard_number} ({path})"
        _require_keys(data, ["frame_idx", "bbox_sm", "fps"], label)
        frame_idx = _contiguous_frame_idx(data["frame_idx"], label)
        if previous_last is not None and int(frame_idx[0]) != previous_last + 1:
            raise ValueError(
                f"HACO metadata archives are not in contiguous order: previous ends at {previous_last}, "
                f"{op.basename(path)} starts at {int(frame_idx[0])}"
            )
        previous_last = int(frame_idx[-1])
        bbox_sm = np.asarray(data["bbox_sm"])
        if bbox_sm.shape != (len(frame_idx), 4) or not np.isfinite(bbox_sm).all():
            raise ValueError(f"{label}:bbox_sm must be finite with shape ({len(frame_idx)}, 4), got {bbox_sm.shape}")
        fps = float(_scalar(data, "fps", label))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"{label}:fps must be finite and positive, got {fps}")
        if first_fps is None:
            first_fps = fps
        elif not _scalars_equal(fps, first_fps):
            raise ValueError(f"{label}:fps={fps} differs from first metadata fps={first_fps}")
        if "img_wh" in data:
            img_wh = np.asarray(data["img_wh"])
            if img_wh.shape != (2,) or not np.issubdtype(img_wh.dtype, np.integer) or np.any(img_wh <= 0):
                raise ValueError(f"{label}:img_wh must be two positive integers, got {img_wh}")
            img_wh = img_wh.astype(np.int64, copy=False)
            if first_img_wh is None:
                first_img_wh = img_wh
            elif not np.array_equal(img_wh, first_img_wh):
                raise ValueError(f"{label}:img_wh={img_wh.tolist()} differs from first metadata archive")
        frames.append(frame_idx)
        boxes.append(bbox_sm.astype(np.float32, copy=False))
        resolved.append(path)

    frame_idx = np.concatenate(frames)
    if not np.array_equal(frame_idx, expected_frames):
        raise ValueError(
            "HACO metadata frame_idx does not exactly match the requested interval: "
            f"got {int(frame_idx[0])}..{int(frame_idx[-1])} ({len(frame_idx)} rows), "
            f"expected {int(expected_frames[0])}..{int(expected_frames[-1])} ({len(expected_frames)} rows)"
        )
    out = {
        "bbox_sm": np.concatenate(boxes),
        "fps": np.asarray(first_fps, dtype=np.float32),
        "haco_meta_archives": np.asarray(resolved),
    }
    if first_img_wh is not None:
        out["img_wh"] = first_img_wh
    return out


def _expect_shape(data, key, shape, label, required=False):
    if key not in data:
        if required:
            raise ValueError(f"{label} is missing key: {key}")
        return None
    value = np.asarray(data[key])
    if value.shape != shape:
        raise ValueError(f"{label}:{key} must have shape {shape}, got {value.shape}")
    return value


def _hawor_focal(data, n, label):
    if "K" in data:
        K = np.asarray(data["K"])
        if K.shape == (3, 3):
            fx = np.full(n, K[0, 0], dtype=np.float64)
            fy = np.full(n, K[1, 1], dtype=np.float64)
        elif K.shape == (n, 3, 3):
            fx, fy = K[:, 0, 0].astype(np.float64), K[:, 1, 1].astype(np.float64)
        else:
            raise ValueError(f"{label}:K must have shape (3, 3) or ({n}, 3, 3), got {K.shape}")
    elif "focal" in data:
        focal = np.asarray(data["focal"], dtype=np.float64)
        if focal.size == 1:
            fx = fy = np.full(n, focal.reshape(-1)[0], dtype=np.float64)
        elif focal.shape == (n,):
            fx = fy = focal
        else:
            raise ValueError(f"{label}:focal must be scalar or shape ({n},), got {focal.shape}")
    else:
        raise ValueError(f"{label} needs K or focal to populate the HACO combined schema")
    if not np.isfinite(fx).all() or not np.isfinite(fy).all() or np.any(fx <= 0) or np.any(fy <= 0):
        raise ValueError(f"{label}:camera focal lengths must be finite and positive")
    if not np.allclose(fx, fy, rtol=1e-5, atol=1e-4):
        raise ValueError(f"{label}:HacoSeqSource requires fx == fy, but HaWoR intrinsics differ")
    return fx.astype(np.float32)


def _load_hawor(path, expected_frames):
    path, data = _load_archive(path, "HaWoR archive")
    label = f"HaWoR archive ({path})"
    _require_keys(data, ["frame_idx", "verts_cam", "joints3d_cam", "faces", "bbox_xyxy"], label)
    input_frames = _contiguous_frame_idx(data["frame_idx"], label)
    n = len(expected_frames)
    if len(input_frames) != n:
        raise ValueError(
            f"{label} has {len(input_frames)} rows, requested source interval has {n} "
            f"({int(expected_frames[0])}..{int(expected_frames[-1])})"
        )
    local_frames = np.arange(n, dtype=np.int64)
    if not np.array_equal(input_frames, local_frames) and not np.array_equal(input_frames, expected_frames):
        raise ValueError(
            f"{label}:frame_idx must be local 0..{n - 1} or the exact requested absolute interval "
            f"{int(expected_frames[0])}..{int(expected_frames[-1])}; got "
            f"{int(input_frames[0])}..{int(input_frames[-1])}"
        )

    verts_cam = _expect_shape(data, "verts_cam", (n, MANO_VERTICES, 3), label, required=True)
    joints_cam = _expect_shape(data, "joints3d_cam", (n, MANO_JOINTS, 3), label, required=True)
    bbox_xyxy = _expect_shape(data, "bbox_xyxy", (n, 4), label, required=True)
    faces = np.asarray(data["faces"])
    if faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError(f"{label}:faces must be an integer (F, 3) array, got {faces.shape} {faces.dtype}")
    if len(faces) == 0 or int(faces.min()) < 0 or int(faces.max()) >= MANO_VERTICES:
        raise ValueError(f"{label}:faces contain indices outside 0..{MANO_VERTICES - 1}")

    optional_shapes = {
        "verts_world": (n, MANO_VERTICES, 3),
        "joints3d_world": (n, MANO_JOINTS, 3),
        "observed": (n,),
        "interpolated": (n,),
        "render_valid": (n,),
        "confidence": (n,),
        "mano_global_orient": (n, 3),
        "mano_hand_pose": (n, 45),
        "mano_betas": (n, 10),
        "cam_R_w2c": (n, 3, 3),
        "cam_t_w2c": (n, 3),
        "cam_R_c2w": (n, 3, 3),
        "cam_t_c2w": (n, 3),
    }
    for key, shape in optional_shapes.items():
        _expect_shape(data, key, shape, label)
    if "img_wh" in data:
        img_wh = np.asarray(data["img_wh"])
        if img_wh.shape != (2,) or not np.issubdtype(img_wh.dtype, np.integer) or np.any(img_wh <= 0):
            raise ValueError(f"{label}:img_wh must be two positive integers, got {img_wh}")
    if "hand_side" in data:
        hand_side = str(np.asarray(data["hand_side"]).reshape(-1)[0])
        if hand_side not in ("left", "right"):
            raise ValueError(f"{label}:hand_side must be 'left' or 'right', got {hand_side!r}")

    finite_tracking = np.isfinite(verts_cam).all(axis=(1, 2)) & np.isfinite(joints_cam).all(axis=(1, 2))
    if "render_valid" in data:
        render_valid = np.asarray(data["render_valid"], dtype=bool)
        bad = np.flatnonzero(render_valid & ~finite_tracking)
        if len(bad):
            raise ValueError(f"{label}:render_valid is true for non-finite tracking row {int(bad[0])}")
    else:
        render_valid = finite_tracking

    wrist_cam = joints_cam[:, 0].astype(np.float32, copy=False)
    payload = dict(data)  # Preserve HaWoR world/camera transforms, provenance, validity, and MANO params.
    payload.update(
        hawor_source_npz=np.asarray(path),
        hawor_frame_idx_input=input_frames,
        hawor_frame_idx_local=local_frames,
        frame_idx=expected_frames,
        verts=(verts_cam - wrist_cam[:, None, :]).astype(np.float32),
        cam_t=wrist_cam,
        focal=_hawor_focal(data, n, label),
        bbox_xyxy=bbox_xyxy.astype(np.float32, copy=False),
        found=render_valid,
        faces=faces.astype(np.int64, copy=False),
        joints3d=(joints_cam - wrist_cam[:, None, :]).astype(np.float32),
    )
    return payload


def _load_object_pose(path, expected_frames):
    if not path:
        return {}
    path, data = _load_archive(path, "Orient Anything archive")
    out = {key: value for key, value in data.items() if key.startswith("object_")}
    if not out:
        raise ValueError(f"Orient Anything archive has no object_* fields: {path}")
    n = len(expected_frames)
    if "object_source_frame" in out:
        source = np.asarray(out["object_source_frame"])
        if source.size != 1 or not np.issubdtype(source.dtype, np.integer):
            raise ValueError(f"Orient Anything object_source_frame must be one integer, got {source}")
        source = int(source.reshape(-1)[0])
        if source not in expected_frames:
            if not 0 <= source < n:
                raise ValueError(
                    f"Orient Anything object_source_frame={source} is neither in the requested absolute "
                    f"interval nor a local row in 0..{n - 1}"
                )
            out["object_source_frame_local"] = np.asarray(source, dtype=np.int64)
            out["object_source_frame"] = np.asarray(expected_frames[source], dtype=np.int64)
    for key, trailing_shape in (("object_bbox_xyxy", (4,)), ("object_rotation_cam", (3, 3))):
        if key not in out:
            continue
        value = np.asarray(out[key])
        if value.shape not in (trailing_shape, (n,) + trailing_shape):
            raise ValueError(
                f"Orient Anything {key} must have shape {trailing_shape} or {(n,) + trailing_shape}, "
                f"got {value.shape}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"Orient Anything {key} contains non-finite values")
    out["object_pose_source_npz"] = np.asarray(path)
    return out


def combine_haco_hawor(contact_shards, hawor_path, output_path, start_frame, end_frame, *,
                       haco_meta=(), orient_path="", haco_checkpoint="",
                       haco_checkpoint_sha256="", haco_stage_b="", haco_model_id=""):
    """Build one HacoSeqSource-compatible archive from HACO contact, HaWoR, and optional object pose."""
    start_frame, end_frame = int(start_frame), int(end_frame)
    if start_frame < 0 or end_frame < start_frame:
        raise ValueError(f"invalid inclusive source-frame interval {start_frame}..{end_frame}")
    expected_frames = np.arange(start_frame, end_frame + 1, dtype=np.int64)

    contact = _load_contact_shards(contact_shards)
    if not np.array_equal(contact["frame_idx"], expected_frames):
        got = contact["frame_idx"]
        raise ValueError(
            "HACO contact frame_idx does not exactly match the requested interval: "
            f"got {int(got[0])}..{int(got[-1])} ({len(got)} rows), "
            f"expected {start_frame}..{end_frame} ({len(expected_frames)} rows)"
        )
    meta = _load_haco_meta(haco_meta, expected_frames)
    hawor = _load_hawor(hawor_path, expected_frames)

    if meta:
        if "img_wh" in meta and "img_wh" in hawor and not np.array_equal(meta["img_wh"], hawor["img_wh"]):
            raise ValueError(
                f"HACO img_wh={meta['img_wh'].tolist()} differs from HaWoR img_wh="
                f"{np.asarray(hawor['img_wh']).tolist()}"
            )
        if "source_fps" in hawor:
            hawor_fps = float(np.asarray(hawor["source_fps"]).reshape(-1)[0])
            if not _scalars_equal(float(meta["fps"]), hawor_fps):
                raise ValueError(f"HACO fps={float(meta['fps'])} differs from HaWoR source_fps={hawor_fps}")
    elif "source_fps" not in hawor:
        raise ValueError("without --haco-meta, the HaWoR archive must provide source_fps")

    payload = dict(hawor)
    payload.update(contact)
    payload.update(meta)
    if bool(haco_checkpoint) != bool(haco_checkpoint_sha256):
        raise ValueError("HACO checkpoint path and SHA-256 must be provided together")
    if haco_checkpoint:
        checkpoint = op.abspath(op.expanduser(str(haco_checkpoint)))
        checksum = str(haco_checkpoint_sha256).lower()
        if not op.isfile(checkpoint):
            raise ValueError(f"HACO checkpoint does not exist: {checkpoint}")
        if len(checksum) != 64:
            raise ValueError("HACO checkpoint SHA-256 must contain 64 hexadecimal characters")
        try:
            int(checksum, 16)
        except ValueError as exc:
            raise ValueError("HACO checkpoint SHA-256 is not hexadecimal") from exc
        payload["haco_checkpoint_path"] = np.asarray(checkpoint)
        payload["haco_checkpoint_sha256"] = np.asarray(checksum)
    if bool(haco_stage_b) != bool(haco_model_id):
        raise ValueError("HACO Stage-B path and model ID must be provided together")
    if haco_stage_b:
        stage_b = op.abspath(op.expanduser(str(haco_stage_b)))
        if not op.isfile(stage_b):
            raise ValueError(f"HACO Stage-B script does not exist: {stage_b}")
        payload["haco_stage_b"] = np.asarray(stage_b)
        payload["haco_model_id"] = np.asarray(str(haco_model_id))
    if "bbox_sm" not in payload:
        payload["bbox_sm"] = np.asarray(payload["bbox_xyxy"], dtype=np.float32)
    if "fps" not in payload:
        fps = float(np.asarray(payload["source_fps"]).reshape(-1)[0])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"HaWoR source_fps must be finite and positive, got {fps}")
        payload["fps"] = np.asarray(fps, dtype=np.float32)
    if "img_wh" not in payload:
        raise ValueError("combined schema requires img_wh from --haco-meta or HaWoR")
    payload.update(_load_object_pose(orient_path, expected_frames))

    n = len(expected_frames)
    per_frame_shapes = {
        "bbox_sm": (n, 4), "contact_prob": (n, MANO_VERTICES), "hand_contact": (n, 2),
        "verts": (n, MANO_VERTICES, 3), "cam_t": (n, 3), "focal": (n,),
        "bbox_xyxy": (n, 4), "found": (n,), "joints3d": (n, MANO_JOINTS, 3),
    }
    for key, shape in per_frame_shapes.items():
        _expect_shape(payload, key, shape, "combined payload", required=True)
    if not np.array_equal(payload["frame_idx"], expected_frames):
        raise ValueError("internal error: combined frame_idx changed during assembly")

    output_path = op.abspath(op.expanduser(str(output_path)))
    output_dir = op.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    temporary = f"{output_path}.tmp-{os.getpid()}.npz"
    try:
        np.savez_compressed(temporary, **payload)
        with np.load(temporary, allow_pickle=True) as saved:
            if not np.array_equal(saved["frame_idx"], expected_frames):
                raise ValueError("saved archive failed frame_idx verification")
            if saved["verts"].shape != (n, MANO_VERTICES, 3):
                raise ValueError(f"saved archive has unexpected verts shape {saved['verts'].shape}")
        os.replace(temporary, output_path)
    finally:
        if op.exists(temporary):
            os.unlink(temporary)

    contact_frames = int((payload["contact_prob"] >= float(payload["thr"])).any(axis=1).sum())
    print(
        f"wrote {output_path}: {n} frames ({start_frame}..{end_frame}), "
        f"HaWoR valid {int(np.asarray(payload['found'], bool).sum())}/{n}, "
        f"contact {contact_frames}/{n}, object pose {'yes' if orient_path else 'no'}"
    )
    print(f"keys: {', '.join(sorted(payload))}")
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Combine ordered HACO contact shards with frame-aligned HaWoR tracking and optional "
            "Orient Anything pose into a HacoSeqSource-compatible NPZ."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--contact-shards", "--haco-contact-shards", dest="contact_shards", nargs="+", required=True,
        help="ordered shard*_contact.npz paths; their frame_idx arrays must tile the requested interval",
    )
    parser.add_argument(
        "--haco-meta", nargs="+", default=(),
        help="optional full Stage-A metadata NPZ or ordered metadata shards (recommended for bbox/fps)",
    )
    parser.add_argument("--hawor", required=True, help="frame-aligned HaWoR NPZ")
    parser.add_argument("--orient", default="", help="optional Orient Anything NPZ; all object_* fields are kept")
    parser.add_argument("--haco-checkpoint", default="", help="HACO checkpoint recorded in output provenance")
    parser.add_argument(
        "--haco-checkpoint-sha256",
        default="",
        help="SHA-256 of --haco-checkpoint recorded in output provenance",
    )
    parser.add_argument("--haco-stage-b", default="", help="Stage-B script recorded in output provenance")
    parser.add_argument("--haco-model-id", default="", help="HACO model ID recorded in output provenance")
    parser.add_argument("--start-frame", type=int, required=True, help="inclusive absolute source frame")
    parser.add_argument("--end-frame", type=int, required=True, help="inclusive absolute source frame")
    parser.add_argument("--output", required=True, help="output combined NPZ")
    args = parser.parse_args(argv)
    try:
        combine_haco_hawor(
            args.contact_shards,
            args.hawor,
            args.output,
            args.start_frame,
            args.end_frame,
            haco_meta=args.haco_meta,
            orient_path=args.orient,
            haco_checkpoint=args.haco_checkpoint,
            haco_checkpoint_sha256=args.haco_checkpoint_sha256,
            haco_stage_b=args.haco_stage_b,
            haco_model_id=args.haco_model_id,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
