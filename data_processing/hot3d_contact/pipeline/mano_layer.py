"""MANO forward pass (PCA hand pose -> 778 posed vertices), read directly from your MANO_*.pkl.

A small standalone implementation of the MANO model (Romero et al., SIGGRAPH Asia 2017): shape blend
shapes, pose-corrective blend shapes and linear blend skinning over the 16-joint kinematic tree. It
follows the conventions HOT3D's annotations are expressed in: PCA hand pose added to the mean pose
(`hands_mean`, i.e. not a flat hand), axis-angle global orientation about the root joint, then a
translation added to every vertex.

The MANO model files are not included; download them from https://mano.is.tue.mpg.de. The pickles
contain `chumpy` arrays; this loader reads their stored values without needing chumpy installed.
"""
import pickle

import numpy as np
import torch


class _ChumpyArray:
    """Stands in for a pickled chumpy array; keeps its stored value (`x`)."""

    def __init__(self, *args, **kwargs):
        self.x = None

    def __setstate__(self, state):
        self.x = state.get("x") if isinstance(state, dict) else None


class _ManoUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _ChumpyArray
        return super().find_class(module, name)


def _dense(value, key):
    if isinstance(value, _ChumpyArray):
        if value.x is None:
            raise ValueError(f"MANO pickle: could not read the value of '{key}'")
        value = value.x
    if hasattr(value, "todense"):          # scipy sparse (J_regressor)
        value = value.todense()
    return np.asarray(value, np.float64)


def load_mano_pkl(path):
    """-> dict of float64 numpy arrays plus integer `faces` and `parents`."""
    with open(path, "rb") as f:
        m = _ManoUnpickler(f, encoding="latin1").load()
    out = {k: _dense(m[k], k) for k in
           ("v_template", "shapedirs", "posedirs", "J_regressor", "weights", "hands_components", "hands_mean")}
    out["faces"] = np.asarray(m["f"], np.int64)
    parents = np.asarray(m["kintree_table"], np.int64)[0].copy()
    parents[0] = -1
    out["parents"] = parents
    return out


def axis_angle_to_matrix(aa):
    """(N, 3) axis-angle -> (N, 3, 3) rotation matrices (Rodrigues' formula)."""
    angle = torch.linalg.norm(aa, dim=1, keepdim=True)                       # (N, 1)
    small = angle < 1e-8
    axis = aa / torch.where(small, torch.ones_like(angle), angle)
    x, y, z = axis.unbind(1)
    zero = torch.zeros_like(x)
    K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], 1).view(-1, 3, 3)
    s = torch.sin(angle)[:, :, None]
    c = torch.cos(angle)[:, :, None]
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device).expand_as(K)
    R = eye + s * K + (1 - c) * (K @ K)
    return torch.where(small[:, :, None], eye, R)


class ManoLayer(torch.nn.Module):
    """MANO with a PCA hand pose. forward() -> (B, 778, 3) vertices, in the units of the model (meters)."""

    def __init__(self, pkl_path, num_pca_comps=15, flat_hand_mean=False, dtype=torch.float32):
        super().__init__()
        d = load_mano_pkl(pkl_path)
        t = lambda a: torch.as_tensor(a, dtype=dtype)
        n_shape = d["shapedirs"].shape[-1]
        self.register_buffer("v_template", t(d["v_template"]))                        # (V, 3)
        self.register_buffer("shapedirs", t(d["shapedirs"]))                          # (V, 3, 10)
        self.register_buffer("posedirs", t(d["posedirs"].reshape(-1, d["posedirs"].shape[-1]).T))  # (135, V*3)
        self.register_buffer("J_regressor", t(d["J_regressor"]))                      # (16, V)
        self.register_buffer("weights", t(d["weights"]))                              # (V, 16)
        self.register_buffer("pca", t(d["hands_components"][:num_pca_comps]))         # (P, 45)
        mean = np.zeros(45) if flat_hand_mean else d["hands_mean"].reshape(45)
        self.register_buffer("hand_mean", t(mean))                                    # (45,)
        self.parents = d["parents"].tolist()
        self.faces = d["faces"]
        self.num_betas = n_shape

    def forward(self, betas, global_orient, hand_pose, transl):
        """betas (B, 10), global_orient (B, 3) axis-angle, hand_pose (B, P) PCA, transl (B, 3)."""
        B = betas.shape[0]
        V = self.v_template.shape[0]
        pose = torch.cat([global_orient, hand_pose @ self.pca + self.hand_mean], 1)        # (B, 48)
        R = axis_angle_to_matrix(pose.reshape(-1, 3)).view(B, -1, 3, 3)                   # (B, 16, 3, 3)
        n_j = R.shape[1]

        v_shaped = self.v_template + torch.einsum("bl,vcl->bvc", betas, self.shapedirs)
        J = torch.einsum("jv,bvc->bjc", self.J_regressor, v_shaped)                       # (B, 16, 3)
        eye = torch.eye(3, dtype=R.dtype, device=R.device)
        pose_feat = (R[:, 1:] - eye).reshape(B, -1)
        v_posed = v_shaped + (pose_feat @ self.posedirs).view(B, V, 3)

        # world transform of every joint along the kinematic tree
        G = [None] * n_j
        for j in range(n_j):
            p = self.parents[j]
            offset = J[:, j] if p < 0 else J[:, j] - J[:, p]
            local = _rigid(R[:, j], offset)
            G[j] = local if p < 0 else G[p] @ local
        G = torch.stack(G, 1)                                                             # (B, 16, 4, 4)
        # skinning transforms: move each joint's rest position to the origin first
        G_rot, G_t = G[..., :3, :3], G[..., :3, 3]
        A = _rigid(G_rot, G_t - torch.einsum("bjik,bjk->bji", G_rot, J))                   # (B, 16, 4, 4)

        T = torch.einsum("vj,bjik->bvik", self.weights, A)                                # (B, V, 4, 4)
        verts = torch.einsum("bvik,bvk->bvi", T[..., :3, :3], v_posed) + T[..., :3, 3]
        return verts + transl[:, None, :]


def _rigid(R, t):
    """(..., 3, 3), (..., 3) -> (..., 4, 4) homogeneous transforms."""
    top = torch.cat([R, t[..., None]], -1)
    bottom = torch.zeros_like(top[..., :1, :])
    bottom[..., 0, 3] = 1
    return torch.cat([top, bottom], -2)
