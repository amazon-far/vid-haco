#!/usr/bin/env python3
"""Build data/base_data/mano_assets.npz from your own copy of MANO_RIGHT.pkl.

The asset file is derived from the MANO model, whose license does not allow redistribution, so it is
not shipped with this repository. Download MANO from https://mano.is.tue.mpg.de (free registration),
then run:

    python tools/make_mano_assets.py --mano data/base_data/human_models/mano/MANO_RIGHT.pkl \
                                     --out data/base_data/mano_assets.npz
"""
import argparse
import pickle

import numpy as np

TIP_VIDS = np.array([744, 320, 443, 554, 671], np.int64)            # thumb, index, middle, ring, pinky tips
KP21_MAP = np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20], np.int64)
FINGER_NAMES = np.array(["palm", "thumb", "index", "middle", "ring", "pinky"])
# MANO joint -> part id (0 palm, 1 thumb, 2 index, 3 middle, 4 ring, 5 pinky).
JOINT_PART = np.array([0, 2, 2, 2, 3, 3, 3, 5, 5, 5, 4, 4, 4, 1, 1, 1], np.int64)


class _Placeholder:
    """Stands in for chumpy objects in the original MANO pickle; none of them are needed here."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _ManoUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _Placeholder
        return super().find_class(module, name)


def wrist_cap(faces):
    """Faces that seal the mesh's one open boundary loop (MANO's wrist), as a fan from one rim vertex.

    A rim edge is a directed face edge whose reverse is in no face. Each cap face runs over its rim edge
    in the reverse direction, so the closed mesh keeps the original faces' consistent winding.
    """
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).tolist()
    present = set(map(tuple, edges))
    succ = {a: b for a, b in edges if (b, a) not in present}
    loop = [next(iter(succ))]
    while succ[loop[-1]] != loop[0]:
        loop.append(succ[loop[-1]])
    if len(loop) != len(succ):
        raise ValueError(f"expected one boundary loop, found {len(succ)} rim edges but a loop of {len(loop)}")
    return np.array([[loop[0], loop[i + 1], loop[i]] for i in range(1, len(loop) - 1)], np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mano", required=True, help="path to MANO_RIGHT.pkl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.mano, "rb") as f:
        m = _ManoUnpickler(f, encoding="latin1").load()
    J = m["J_regressor"]
    J = np.asarray(J.todense() if hasattr(J, "todense") else J, np.float32)
    weights = np.asarray(m["weights"], np.float32)
    faces = np.asarray(m["f"], np.int64)
    np.savez(a.out,
             J_regressor=J, lbs_weights=weights, faces=faces,
             watertight_faces=np.concatenate([faces, wrist_cap(faces)]),
             vert_finger=JOINT_PART[weights.argmax(1)], tip_vids=TIP_VIDS, kp21_map=KP21_MAP,
             finger_names=FINGER_NAMES, v_template=np.asarray(m["v_template"], np.float32),
             source=np.array("MANO_RIGHT.pkl"))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
