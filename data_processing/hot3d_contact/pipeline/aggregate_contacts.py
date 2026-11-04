#!/usr/bin/env python3
"""Aggregate per-sequence HOT3D contact npz into HACO-style dataset artifacts.

Outputs under --out:
  contact_data_all.npy   (N,778) uint8   all hand-instances (both hands, contact frames+noncontact)
  contact_means.npy      (778,) float64  per-vertex contact frequency (HACO 'contact_means')
  manifest.csv           row -> seq,hand,timestamp_ns,n_contact
  dataset_stats.json     summary stats (per split/headset/object)
"""
import os, sys, glob, json, csv
import numpy as np
from collections import defaultdict

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/hot3d_contact/contact_data")
    ap.add_argument("--out", default="data/hot3d_contact/contact_data")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.root, "*", "*.npz")))
    print(f"aggregating {len(files)} sequence npz")

    rows_contact = []          # (778,) uint8 per hand-instance (present frames only)
    manifest = []
    stats = dict(n_sequences=0, n_hand_instances=0, n_contact_instances=0,
                 per_headset=defaultdict(lambda: dict(seqs=0, frames=0, hand_inst=0, contact_inst=0)),
                 per_object_contact=defaultdict(int), per_object_frames=defaultdict(int))
    vsum = np.zeros(778, np.float64); vn = 0
    bad = 0
    for f in files:
        try:
            d = np.load(f, allow_pickle=True)
        except Exception as e:
            bad += 1; print("  bad", f, e); continue
        meta = json.loads(str(d["meta"])); seq = meta["seq"]; head = meta.get("headset") or "?"
        stats["n_sequences"] += 1
        hs = stats["per_headset"][head]; hs["seqs"] += 1; hs["frames"] += int(meta["n_frames"])
        names = list(d["object_names"])
        ts = d["timestamps_ns"]
        for side, hk in (("left", "L"), ("right", "R")):
            pres = d[f"{side}_present"]; C = d[f"{side}_contact"]; near = d[f"{side}_nearest_obj"]
            idx = np.where(pres)[0]
            for i in idx:
                c = C[i]
                rows_contact.append(c)
                nc = int(c.sum())
                manifest.append((seq, side, int(ts[i]), nc))
                vsum += c; vn += 1
                stats["n_hand_instances"] += 1; hs["hand_inst"] += 1
                if nc > 0:
                    stats["n_contact_instances"] += 1; hs["contact_inst"] += 1
                    for g in np.unique(near[i][near[i] >= 0]):
                        stats["per_object_contact"][names[int(g)]] += int((near[i] == g).sum())
                        stats["per_object_frames"][names[int(g)]] += 1
    if not rows_contact:
        print("no data"); return
    allc = np.stack(rows_contact).astype(np.uint8)
    means = vsum / max(1, vn)
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "contact_data_all.npy"), allc)
    np.save(os.path.join(a.out, "contact_means.npy"), means)
    with open(os.path.join(a.out, "manifest.csv"), "w", newline="") as fo:
        w = csv.writer(fo); w.writerow(["seq", "hand", "timestamp_ns", "n_contact_verts"])
        w.writerows(manifest)
    # finalize stats
    stats["per_headset"] = {k: v for k, v in stats["per_headset"].items()}
    stats["per_object_contact"] = dict(sorted(stats["per_object_contact"].items(), key=lambda x: -x[1]))
    stats["per_object_frames"] = dict(sorted(stats["per_object_frames"].items(), key=lambda x: -x[1]))
    stats["contact_instance_frac"] = stats["n_contact_instances"] / max(1, stats["n_hand_instances"])
    stats["mean_contact_verts_over_contact_inst"] = float(allc[allc.sum(1) > 0].sum(1).mean()) if (allc.sum(1) > 0).any() else 0
    stats["overall_vertex_contact_freq_mean"] = float(means.mean())
    stats["aggregate_shape"] = list(allc.shape)
    stats["bad_files"] = bad
    json.dump(stats, open(os.path.join(a.out, "dataset_stats.json"), "w"), indent=1)
    print(f"DONE: aggregate {allc.shape}, contact_frac {stats['contact_instance_frac']:.3f}, "
          f"mean contact/inst {stats['mean_contact_verts_over_contact_inst']:.1f}")
    print("top objects by contact frames:", list(stats["per_object_frames"].items())[:8])

if __name__ == "__main__":
    main()
