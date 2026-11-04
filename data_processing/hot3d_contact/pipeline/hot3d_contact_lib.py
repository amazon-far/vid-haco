"""HOT3D dense hand-contact labeling — core library.

Labeling rule of HACO (Jung & Lee, arXiv 2505.11152): a MANO hand vertex is in contact when its
Euclidean distance to the nearest point on an object SURFACE is below c_thres (1 cm for
hand-object) -> binary per-MANO-vertex (778,) contact label.

Hand vertices come from HOT3D MANO annotations, posed with the same conventions as HOT3D's own
data_loaders/mano_layer.py (PCA hand pose, 15 components, mean-pose offset; see mano_layer.py).
Object meshes come from the HOT3D object library (.glb), posed to world via
dynamic_objects.csv (t_wo, q_wo : world-from-object).
"""
import os, json, glob
import numpy as np
import torch
import trimesh
from trimesh.proximity import ProximityQuery
from scipy.spatial.transform import Rotation as Rot

from mano_layer import ManoLayer

MANO_DIR = "data/base_data/human_models/mano"
C_THRES = 0.01  # 1 cm, the hand-object threshold used by HACO
N_VERT = 778
LEFT, RIGHT = "0", "1"   # HOT3D handedness keys

# ----------------------------------------------------------------------------- MANO
class ManoWrap:
    """Left and right MANO layers with HOT3D's conventions (PCA pose, 15 comps)."""
    def __init__(self, device="cpu"):
        self.device = device
        self.layer = {
            RIGHT: ManoLayer(os.path.join(MANO_DIR, "MANO_RIGHT.pkl"), num_pca_comps=15).to(device),
            LEFT:  ManoLayer(os.path.join(MANO_DIR, "MANO_LEFT.pkl"), num_pca_comps=15).to(device),
        }
        # MANO_LEFT.pkl's shape directions carry the right hand's x components instead of mirrored ones;
        # mirror them, as HOT3D's mano_layer.py does
        l, r = self.layer[LEFT], self.layer[RIGHT]
        if torch.sum(torch.abs(l.shapedirs[:, 0, :] - r.shapedirs[:, 0, :])) < 1:
            l.shapedirs[:, 0, :] *= -1
        self.faces = {k: self.layer[k].faces.astype(np.int64) for k in (LEFT, RIGHT)}

    @torch.no_grad()
    def forward(self, hand_key, betas, pose15, wrist_q_wxyz, wrist_t, batch=True):
        """betas (B,10), pose15 (B,15), wrist_q_wxyz (B,4), wrist_t (B,3) -> verts (B,778,3) world frame.
        Replicates ManoHandDataProvider.get_hand_mesh_vertices: wrist SE3 -> axis-angle global_orient + transl."""
        betas = torch.as_tensor(betas, dtype=torch.float32, device=self.device)
        pose15 = torch.as_tensor(pose15, dtype=torch.float32, device=self.device)
        q = np.asarray(wrist_q_wxyz, dtype=np.float64).reshape(-1, 4)
        # scipy wants xyzw; HOT3D stores wxyz
        aa = Rot.from_quat(q[:, [1, 2, 3, 0]]).as_rotvec().astype(np.float32)  # (B,3) axis-angle
        global_orient = torch.as_tensor(aa, device=self.device)
        transl = torch.as_tensor(np.asarray(wrist_t, np.float32).reshape(-1, 3), device=self.device)
        v = self.layer[hand_key](betas, global_orient, pose15, transl)  # (B,778,3), world frame (meters)
        return v if batch else v[0]

# ----------------------------------------------------------------------------- parsing
def load_hand_traj(seq_dir):
    """Return dict: timestamp_ns -> {hand_key: {pose[15], q_wxyz[4], t[3], betas[10]}}."""
    path = os.path.join(seq_dir, "mano_hand_pose_trajectory.jsonl")
    frames = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            ts = d["timestamp_ns"]
            hp = {}
            for hk in (LEFT, RIGHT):
                if hk in d["hand_poses"] and "pose" in d["hand_poses"][hk]:
                    e = d["hand_poses"][hk]
                    hp[hk] = dict(pose=e["pose"], q_wxyz=e["wrist_xform"]["q_wxyz"],
                                  t=e["wrist_xform"]["t_xyz"], betas=e["betas"])
            if hp:
                frames[ts] = hp
    return frames

def load_object_traj(seq_dir):
    """Return dict: timestamp_ns -> list[(obj_uid, 4x4 world_from_object)]."""
    import csv
    path = os.path.join(seq_dir, "dynamic_objects.csv")
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            ts = int(row["timestamp[ns]"])
            t = np.array([float(row["t_wo_x[m]"]), float(row["t_wo_y[m]"]), float(row["t_wo_z[m]"])])
            q = [float(row["q_wo_w"]), float(row["q_wo_x"]), float(row["q_wo_y"]), float(row["q_wo_z"])]
            R = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t
            out.setdefault(ts, []).append((row["object_uid"], M))
    return out

# ----------------------------------------------------------------------------- objects
from scipy.spatial import cKDTree

# Surface-sample density: aim for ~1.2 mm linear spacing so the KD-tree distance
# approximates the true point-to-surface distance to well under the 1 cm threshold.
SAMPLE_SPACING = 0.0012      # meters between surface samples (target)
SAMPLE_MIN = 40000
SAMPLE_MAX = 250000


class ObjectLibrary:
    def __init__(self, assets_dir, seed=0):
        self.assets_dir = assets_dir
        self.seed = seed
        self._mesh = {}   # uid -> trimesh.Trimesh (canonical frame)
        self._pq = {}     # uid -> ProximityQuery (exact; used only for validation)
        self._tree = {}   # uid -> cKDTree over dense surface samples (canonical frame)
        self._bbox = {}   # uid -> (2,3) canonical aabb
        self.instance = json.load(open(os.path.join(assets_dir, "instance.json")))

    def mesh(self, uid):
        if uid not in self._mesh:
            m = trimesh.load(os.path.join(self.assets_dir, f"{uid}.glb"),
                             force="mesh", process=False)
            self._mesh[uid] = m
            self._bbox[uid] = np.array([m.vertices.min(0), m.vertices.max(0)])
        return self._mesh[uid]

    def pq(self, uid):
        """Exact trimesh ProximityQuery (slow; validation only)."""
        if uid not in self._pq:
            self._pq[uid] = ProximityQuery(self.mesh(uid))
        return self._pq[uid]

    def tree(self, uid):
        """cKDTree over dense surface samples + mesh vertices, in canonical frame."""
        if uid not in self._tree:
            m = self.mesh(uid)
            area = float(m.area)
            n = int(np.clip(area / (SAMPLE_SPACING ** 2), SAMPLE_MIN, SAMPLE_MAX))
            rng = np.random.RandomState(self.seed)
            try:
                pts, _ = trimesh.sample.sample_surface(m, n, seed=self.seed) \
                    if 'seed' in trimesh.sample.sample_surface.__code__.co_varnames \
                    else trimesh.sample.sample_surface(m, n)
            except Exception:
                pts = m.vertices
            pts = np.vstack([np.asarray(pts, np.float64), np.asarray(m.vertices, np.float64)])
            self._tree[uid] = cKDTree(pts)
        return self._tree[uid]

    def bbox(self, uid):
        self.mesh(uid)
        return self._bbox[uid]

    def name(self, uid):
        return self.instance.get(str(uid), {}).get("instance_name", str(uid))


def _aabb_gap(a, b):
    """Euclidean gap between two AABBs (each (2,3): [min;max]); 0 if overlapping."""
    d = np.maximum(0.0, np.maximum(a[0] - b[1], b[0] - a[1]))
    return float(np.sqrt((d * d).sum()))


def contact_multi_object(hand_verts, objs, lib, c_thres=C_THRES):
    """Contact rule (surface distance < c_thres) against all present objects (hand-object).

    Distance is rigid-invariant, so instead of posing each object to world and
    rebuilding its rtree every frame, we build each object's ProximityQuery ONCE
    in its canonical frame (cached in `lib`) and transform the hand into that
    frame. An AABB prefilter skips objects that cannot possibly contact.

    hand_verts (778,3) world frame.
    objs: list[(uid, 4x4 world_from_object)].
    Returns: contact(778) uint8, min_dist(778) float32, nearest_obj_local_idx(778) int8,
             kept_uids (list, index = the nearest_obj values).
    """
    n = len(hand_verts)
    min_dist = np.full(n, np.inf, np.float32)
    nearest = np.full(n, -1, np.int16)
    hb = np.array([hand_verts.min(0), hand_verts.max(0)])
    kept = []
    for uid, M in objs:
        R, t = M[:3, :3], M[:3, 3]
        try:
            cb = lib.bbox(uid)
        except Exception:
            continue
        # object canonical bbox -> world aabb (transform 8 corners)
        corners = np.array([[cb[i, 0], cb[j, 1], cb[k, 2]]
                            for i in (0, 1) for j in (0, 1) for k in (0, 1)])
        cw = corners @ R.T + t
        ob = np.array([cw.min(0), cw.max(0)])
        if _aabb_gap(hb, ob) > c_thres + 1e-3:
            continue
        idx = len(kept)
        kept.append(uid)
        hv_obj = (hand_verts - t) @ R   # world -> object frame
        dist, _ = lib.tree(uid).query(hv_obj, k=1, workers=1)
        dist = dist.astype(np.float32)
        upd = dist < min_dist
        min_dist[upd] = dist[upd]
        nearest[upd] = idx
    contact = (min_dist < c_thres).astype(np.uint8)
    nearest[contact == 0] = -1
    return contact, min_dist, nearest.astype(np.int16), kept

# ----------------------------------------------------------------------------- contact
def build_world_object_mesh(objs, lib):
    """Concatenate present objects into one world-frame trimesh; return (mesh, tri2objidx list)."""
    vlist, flist, tri_obj = [], [], []
    voff = 0
    kept = []
    for i, (uid, M) in enumerate(objs):
        try:
            v, f = lib.get(uid)
        except Exception:
            continue
        vw = v @ M[:3, :3].T + M[:3, 3]
        vlist.append(vw); flist.append(f + voff)
        tri_obj.append(np.full(len(f), len(kept), np.int32))
        voff += len(vw); kept.append(uid)
    if not vlist:
        return None, None, []
    mesh = trimesh.Trimesh(np.vstack(vlist), np.vstack(flist), process=False)
    return mesh, np.concatenate(tri_obj), kept

def contact_for_hand(hand_verts, obj_mesh, tri_obj, c_thres=C_THRES):
    """Exact contact rule. hand_verts (778,3), obj_mesh trimesh -> (contact(778) uint8, dist(778),
    nearest_obj_idx(778) int8, closest_pts(778,3))."""
    pq = ProximityQuery(obj_mesh)
    closest, dist, tri = pq.on_surface(hand_verts.astype(np.float32))
    contact = (dist < c_thres).astype(np.uint8)
    nearest_obj = tri_obj[tri].astype(np.int8) if tri_obj is not None else np.full(len(dist), -1, np.int8)
    nearest_obj[contact == 0] = -1
    return contact, dist.astype(np.float32), nearest_obj, closest.astype(np.float32)
