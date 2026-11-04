#!/usr/bin/env python3
"""Compute HACO-style dense hand-contact maps for all HOT3D sequences.

Per sequence -> one compressed .npz:
  timestamps_ns [T] int64
  {left,right}_present [T] bool
  {left,right}_contact [T,778] uint8      # binary contact @ 1cm, HACO rule  (the deliverable)
  {left,right}_min_dist [T,778] float16   # hand-vertex -> nearest object surface (m), capped 0.2
  {left,right}_nearest_obj [T,778] int8   # global object index (-1 if no contact)
  object_uids [O] str, object_names [O] str
  meta (json string): seq, headset, c_thres, mano cfg, stats

Parallel across sequences with a process pool. Thread libs pinned to 1 to avoid
oversubscription. Same per-vertex (778,) representation as HACO's labels.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys, json, time, argparse, traceback
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ASSETS = "data/hot3d_dataset/assets"
DIST_CAP = 0.2  # meters; stored min_dist capped for float16

_G = {}  # per-worker globals (lazy)

def _init():
    import torch
    torch.set_num_threads(1)
    import hot3d_contact_lib as L
    _G["L"] = L
    _G["mano"] = L.ManoWrap()
    _G["lib"] = L.ObjectLibrary(ASSETS)

def process_sequence(args):
    seq_dir, out_path, c_thres = args
    try:
        if "L" not in _G:
            _init()
        L = _G["L"]; mano = _G["mano"]; lib = _G["lib"]
        seq = os.path.basename(seq_dir.rstrip("/"))
        meta_seq = {}
        mp = os.path.join(seq_dir, "metadata.json")
        if os.path.exists(mp):
            meta_seq = json.load(open(mp))
        hands = L.load_hand_traj(seq_dir)
        if not hands:
            return dict(seq=seq, status="skip_no_hand", n_frames=0)
        objs = L.load_object_traj(seq_dir)
        ts_all = sorted(hands.keys())
        T = len(ts_all)
        ti = {t: i for i, t in enumerate(ts_all)}

        # global object index for this sequence
        obj_uids = sorted({u for lst in objs.values() for u, _ in lst})
        uid2g = {u: i for i, u in enumerate(obj_uids)}

        out = {"timestamps_ns": np.array(ts_all, np.int64)}
        stats = {}
        for hk, side in ((L.LEFT, "left"), (L.RIGHT, "right")):
            idx = [t for t in ts_all if hk in hands[t]]
            present = np.zeros(T, bool)
            contact = np.zeros((T, L.N_VERT), np.uint8)
            mindist = np.full((T, L.N_VERT), DIST_CAP, np.float16)
            nearest = np.full((T, L.N_VERT), -1, np.int8)
            if idx:
                arr = lambda k: np.array([hands[t][hk][k] for t in idx], np.float32)
                V = mano.forward(hk, arr("betas"), arr("pose"), arr("q_wxyz"), arr("t"),
                                 batch=True).numpy()
                for j, t in enumerate(idx):
                    i = ti[t]; present[i] = True
                    c, d, near, kept = L.contact_multi_object(V[j], objs.get(t, []), lib, c_thres)
                    contact[i] = c
                    mindist[i] = np.minimum(d, DIST_CAP).astype(np.float16)
                    # map local kept idx -> global object index
                    gn = np.full(L.N_VERT, -1, np.int8)
                    hit = near >= 0
                    if hit.any():
                        loc2g = np.array([uid2g[kept[k]] for k in range(len(kept))], np.int8) \
                            if kept else np.zeros(0, np.int8)
                        gn[hit] = loc2g[near[hit]]
                    nearest[i] = gn
            out[f"{side}_present"] = present
            out[f"{side}_contact"] = contact
            out[f"{side}_min_dist"] = mindist
            out[f"{side}_nearest_obj"] = nearest
            npr = int(present.sum())
            cv = int(contact.sum())
            stats[side] = dict(present_frames=npr,
                               mean_contact_verts=(cv / npr if npr else 0.0),
                               contact_frames=int((contact.sum(1) > 0).sum()))
        out["object_uids"] = np.array(obj_uids, dtype=object)
        out["object_names"] = np.array([lib.name(u) for u in obj_uids], dtype=object)
        meta = dict(seq=seq, headset=meta_seq.get("headset"), c_thres=c_thres,
                    n_frames=T, participant=meta_seq.get("participant_id"),
                    mano="MANO use_pca=True ncomps=15 (HOT3D convention)",
                    method="HACO contact rule (surface dist < c_thres); cKDTree(dense surf samples) approx of trimesh on_surface",
                    stats=stats)
        out["meta"] = json.dumps(meta)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, **out)
        return dict(seq=seq, status="ok", n_frames=T, **stats)
    except Exception as e:
        return dict(seq=os.path.basename(seq_dir.rstrip("/")), status="error",
                    err=f"{e}\n{traceback.format_exc()[-800:]}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["data/hot3d_dataset/aria",
                                                   "data/hot3d_dataset/quest"])
    ap.add_argument("--out", default="data/hot3d_contact/contact_data")
    ap.add_argument("--workers", type=int, default=160)
    ap.add_argument("--c_thres", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    seqs = []
    for root in a.roots:
        head = os.path.basename(root)
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "mano_hand_pose_trajectory.jsonl")):
                seqs.append((d, os.path.join(a.out, head, f"{name}.npz"), a.c_thres))
    if a.limit:
        seqs = seqs[:a.limit]
    # skip already-done
    todo = [s for s in seqs if not os.path.exists(s[1])]
    print(f"[contacts] {len(seqs)} seqs, {len(todo)} to do, {a.workers} workers", flush=True)

    t0 = time.time()
    ok = err = skip = 0
    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(process_sequence, s): s for s in todo}
        for n, f in enumerate(as_completed(futs), 1):
            r = f.result(); results.append(r)
            st = r["status"]
            if st == "ok": ok += 1
            elif st == "error": err += 1; print(f"  ERROR {r['seq']}: {r.get('err','')[:300]}", flush=True)
            else: skip += 1
            if n % 20 == 0 or st == "error":
                el = time.time() - t0
                print(f"[{n}/{len(todo)}] ok={ok} skip={skip} err={err} | {el/60:.1f}m "
                      f"ETA {(len(todo)-n)/(n/el)/60:.1f}m", flush=True)
    json.dump(results, open(os.path.join(a.out, "_run_summary.json"), "w"), indent=1)
    print(f"[contacts] DONE ok={ok} skip={skip} err={err} in {(time.time()-t0)/60:.1f}m", flush=True)

if __name__ == "__main__":
    main()
