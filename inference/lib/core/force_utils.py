"""Single source of truth for the fingertip-force target transform and its inverse.

Training, Stage-B inference and the eval scripts must agree exactly on how newtons map to the value
the network regresses, so all of it lives here.

PER-CHANNEL p90 NORMALISATION (current scheme)
----------------------------------------------
Each of the 6 channels is normalised by ITS OWN 90th percentile, which is mapped to 20, then the
result is log1p'd (see "Why log1p" below):

    target_c = log1p(clip(newtons_c * scale_c, 0, 20))     scale_c = 20 / p90_c

Constants live in data/FEEL/force_norm.npz (computed once from the FEEL TRAIN split, frozen so that
inference and training cannot drift apart). Per-channel matters because the channels are wildly
different -- p90 ranges from 0.25 N (ring) to 0.92 N (palm), so one global scale would leave some
fingers in the noise floor and saturate others:

    thumb 0.803 N -> x24.9 | index 0.383 -> x52.2 | middle 0.339 -> x59.0
    ring  0.254 N -> x78.9 | pinky 0.464 -> x43.1 | palm   0.923 -> x21.7

!! p90 is taken over NONZERO readings. Over ALL readings the palm channel is 90.1% zeros, so its p90
   is exactly 0.0000 and the scale would be a divide-by-zero. Using the nonzero p90 also matches the
   intent ("the p90 of the force data"), and clips only 2.58% of readings at the 20 ceiling.

The normalised target is then passed through log1p, giving a [0, 3.04] range:

    target_c = log1p(clip(newtons_c * scale_c, 0, 20))

Why log1p and not the raw linear 0..20 (which an earlier revision used):
  * LOSS BALANCE. Against a zero-init head the magnitude SmoothL1 is ~6.48 on the linear target vs
    the gate's BCE of ~0.69 -- a ~9x imbalance that starves the gate (measured: gate AUC fell
    0.73 -> 0.64 under the linear target). In log space the same term is ~1.43, i.e. ~2.1x, so the
    two heads train together instead of one drowning the other.
  * RELATIVE ERROR. Force spans two orders of magnitude; a 0.02 N miss means something very
    different at 0.05 N than at 0.5 N. log space weights them proportionally.
  * NO DECODE BIAS. The usual objection (E[y] != expm1(E[log1p y])) applies to a MEAN-seeking loss.
    SmoothL1 is MEDIAN-seeking in its linear region, and the median commutes with a monotone
    transform, so expm1-decoding a median fit is unbiased. (This is also why the objection did bite
    the very first version: there the conditional median was 0 because zeros were in the mix -- now
    the magnitude head only ever sees LOADED readings.)

HURDLE (two-part) HEAD
----------------------
FEEL is ~75% zeros, and a single regressor on that is minimised by predicting ~0 -- which is exactly
what the first model did. So the head is split:

    E[y] = P(y > tau) * E[y | y > tau]
           \__gate__/   \__magnitude__/

The gate trains on EVERY force-supervised reading; the magnitude head trains ONLY on readings above
tau, so the mass of zeros cannot drag it down. `decode_force` ALWAYS returns the PRODUCT -- the
magnitude head is meaningless alone, because having seen only positives it will happily emit force on
a resting hand. tau = 0.01 N (not literally >0) keeps the gate's job well posed.
"""
import os
import numpy as np
import torch

N_FORCE = 6
FORCE_TAU_N = 0.01                 # hurdle threshold, in NEWTONS
FORCE_TARGET_AT_P90 = 20.0         # per-channel p90 maps here
FORCE_CEIL = 20.0                  # ceiling in NORMALISED units
ENCODING_VERSION = 'perchannel_p90_20_log1p_v3'   # stamped into Stage-B npz; see check_encoding()

_NORM_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'FEEL', 'force_norm.npz')
_SCALE = None


def _scale():
    """Per-channel scale factors [6], loaded once from the frozen constants file."""
    global _SCALE
    if _SCALE is None:
        p = os.path.abspath(_NORM_PATH)
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"force normalisation constants missing: {p}\n"
                "Regenerate with the p90 snippet in the FEEL notes (p90 of NONZERO readings per "
                "channel over the FEEL train split).")
        z = np.load(p)
        _SCALE = np.asarray(z['scale'], np.float32).reshape(-1)
        assert _SCALE.shape == (N_FORCE,), f"bad scale shape {_SCALE.shape}"
    return _SCALE


def scale_np():
    return _scale().copy()


def encode_target(newtons):
    """newtons [..., 6] -> log1p of the per-channel normalised force, in [0, 3.04].

    This is exactly what the magnitude head regresses; decode_force is its inverse.
    """
    s = _scale()
    if isinstance(newtons, torch.Tensor):
        st = torch.as_tensor(s, dtype=newtons.dtype, device=newtons.device)
        return torch.log1p(torch.clamp(newtons * st, min=0.0, max=FORCE_CEIL))
    return np.log1p(np.clip(np.asarray(newtons, np.float64) * s, 0.0, FORCE_CEIL))


def decode_force(mag_norm, presence_logit=None):
    """Inverse: head output(s) -> NEWTONS.

    mag_norm       : magnitude head output, log1p of normalised units
    presence_logit : hurdle gate logits. If given the result is gate * magnitude -- ALWAYS use this
                     for anything user-facing. Without it (a pre-hurdle checkpoint) the bare
                     magnitude is returned, which is the old pure-regression behaviour.
    """
    s = _scale()
    if isinstance(mag_norm, torch.Tensor):
        st = torch.as_tensor(s, dtype=mag_norm.dtype, device=mag_norm.device)
        mag = torch.clamp(torch.expm1(mag_norm), min=0.0, max=FORCE_CEIL) / st
        return mag if presence_logit is None else torch.sigmoid(presence_logit) * mag
    mag = np.clip(np.expm1(np.asarray(mag_norm, np.float64)), 0.0, FORCE_CEIL) / s
    if presence_logit is None:
        return mag
    p = 1.0 / (1.0 + np.exp(-np.asarray(presence_logit, np.float64)))
    return p * mag


def check_encoding(stamp, where=''):
    """Fail loudly when an npz was written under a different force encoding.

    The encoding has already changed twice (log1p-newtons -> log1p(10x, ceil 10 N) -> per-channel
    p90). Decoding an old file with a new decoder silently rescales the newtons instead of erroring,
    which is exactly the kind of bug that reaches a figure. So Stage B stamps the version and every
    consumer checks it.
    """
    got = str(np.asarray(stamp).reshape(-1)[0]) if stamp is not None else '(none)'
    if got != ENCODING_VERSION:
        raise ValueError(
            f"force encoding mismatch{(' in ' + where) if where else ''}: file says '{got}', this "
            f"code is '{ENCODING_VERSION}'. Re-run Stage B with the current checkpoint/decoder.")
