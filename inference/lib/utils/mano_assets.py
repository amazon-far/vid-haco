"""
The MANO arrays the stages need, read from `data/base_data/mano_assets.npz`.

That file is derived from your own copy of MANO_RIGHT.pkl by `tools/make_mano_assets.py` (MANO's license
does not allow redistributing it). It holds the template mesh, its faces with and without the wrist
opening closed, the joint regressor, the skinning weights and a few vertex/joint index tables.
"""
import os

import numpy as np

NUM_VERTS = 778
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'data', 'base_data', 'mano_assets.npz')


def load(path=DEFAULT_PATH):
    """-> dict of arrays (J_regressor, faces, watertight_faces, v_template, lbs_weights, ...)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found — build it with: python tools/make_mano_assets.py "
            f"--mano data/base_data/human_models/mano/MANO_RIGHT.pkl --out data/base_data/mano_assets.npz")
    with np.load(path, allow_pickle=False) as z:
        assets = {k: z[k] for k in z.files}
    if assets['v_template'].shape[0] != NUM_VERTS:
        raise ValueError(f"{path}: expected a {NUM_VERTS}-vertex MANO hand, "
                         f"got {assets['v_template'].shape[0]} vertices")
    return assets
