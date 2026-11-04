"""
The per-frame contact mask the renderer draws: thresholded probabilities, smoothed over time with a
causal EMA, then cleaned of contact islands too small to be real.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def ema(state, mask, alpha=0.8):
    """One step of the causal EMA over masks; `state` is None on the first frame."""
    mask = np.asarray(mask, np.float32)
    return mask if state is None else alpha * state + (1.0 - alpha) * mask


def drop_small_components(mask, faces, min_size=20):
    """Keep only the contact regions of at least `min_size` vertices.

    Two contact vertices are connected when they share a mesh edge; pass the watertight faces so the
    regions on either side of the wrist opening count as touching.
    """
    mask = np.asarray(mask, bool)
    keep = np.zeros_like(mask)
    if not mask.any():
        return keep
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = edges[mask[edges[:, 0]] & mask[edges[:, 1]]]
    n = mask.size
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    _, label = connected_components(graph, directed=False)
    size = np.bincount(label[mask], minlength=label.max() + 1)
    keep[mask] = size[label[mask]] >= min_size
    return keep
