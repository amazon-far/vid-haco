#!/usr/bin/env python3
"""Build a HOT3D contact-map visualization gallery for review.

  viz/contact_frequency_hand.png     dataset-wide per-vertex contact heatmap
  viz/frames/*.png                   curated 2-panel per-frame contact renders (diverse objects/hands)
  viz/contact_sheet.png              grid of the above
  viz/gifs/*.gif                     grasp contact-evolution clips
  viz/gallery_index.json             what was rendered
"""
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import sys, glob, json, numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hot3d_contact_lib as L
import viz_contacts as VZ

OUT = "data/hot3d_contact/viz"
DATA = "data/hot3d_contact/contact_data"
DS = {"aria": "data/hot3d_dataset/aria", "quest": "data/hot3d_dataset/quest"}

def load_all():
    items = []
    for f in sorted(glob.glob(os.path.join(DATA, "*", "*.npz"))):
        try:
            d = np.load(f, allow_pickle=True)
        except Exception:
            continue
        head = os.path.basename(os.path.dirname(f)); seq = os.path.basename(f)[:-4]
        items.append((head, seq, f, d))
    return items

def curate(items, per_object=3, cap_per_seq=2):
    """Pick diverse high-contact frames covering many object categories and both hands."""
    cand = []
    for head, seq, f, d in items:
        names = list(d["object_names"]) if len(d["object_names"]) else []
        ts = d["timestamps_ns"]
        for side, hk in (("left", L.LEFT), ("right", L.RIGHT)):
            C = d[f"{side}_contact"]; near = d[f"{side}_nearest_obj"]; pres = d[f"{side}_present"]
            nc = C.sum(1)
            good = np.where((nc >= 25) & pres)[0]
            for i in good:
                on = near[i][near[i] >= 0]
                obj = names[int(np.bincount(on).argmax())] if len(on) and names else "?"
                cand.append((int(nc[i]), head, seq, side, hk, int(ts[i]), obj))
    # bucket by object, take top per object with per-seq cap
    by_obj = {}
    for c in sorted(cand, reverse=True):
        by_obj.setdefault(c[6], []).append(c)
    picks = []
    for obj, lst in sorted(by_obj.items(), key=lambda x: -len(x[1])):
        seen = {}
        for c in lst:
            if seen.get(c[2], 0) >= cap_per_seq: continue
            seen[c[2]] = seen.get(c[2], 0) + 1
            picks.append(c)
            if sum(1 for p in picks if p[6] == obj) >= per_object: break
    picks.sort(reverse=True)
    return picks

def main():
    os.makedirs(os.path.join(OUT, "frames"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "gifs"), exist_ok=True)
    mano = L.ManoWrap(); lib = L.ObjectLibrary("data/hot3d_dataset/assets")

    # 1) contact-frequency hand heatmap
    means_p = os.path.join(DATA, "contact_means.npy")
    if os.path.exists(means_p):
        VZ.render_template_heatmap(mano, np.load(means_p),
                                   os.path.join(OUT, "contact_frequency_hand.png"))
        print("rendered contact_frequency_hand.png")

    items = load_all()
    print(f"loaded {len(items)} sequences")
    picks = curate(items)
    print(f"curated {len(picks)} frames covering objects:",
          sorted(set(p[6] for p in picks)))

    index = {"frames": [], "gifs": []}
    tiles = []
    for k, (nc, head, seq, side, hk, ts, obj) in enumerate(picks[:30]):
        seq_dir = os.path.join(DS[head], seq)
        d = np.load(os.path.join(DATA, head, f"{seq}.npz"), allow_pickle=True)
        try:
            img = VZ.render_frame_2panel(mano, lib, seq_dir, d, ts, hk)
        except Exception as e:
            print("  frame fail", seq, e); continue
        name = f"{k:02d}_{obj}_{seq}_{side}_{nc}v.png"
        Image.fromarray(img).save(os.path.join(OUT, "frames", name))
        index["frames"].append(name); tiles.append(img)
    print(f"rendered {len(tiles)} frames")

    # contact sheet grid
    if tiles:
        cols = 2; rows = (len(tiles) + cols - 1) // cols
        h, w = tiles[0].shape[:2]
        sheet = np.full((rows*h, cols*w, 3), 255, np.uint8)
        for i, t in enumerate(tiles):
            r, c = divmod(i, cols)
            sheet[r*h:r*h+t.shape[0], c*w:c*w+t.shape[1]] = t
        Image.fromarray(sheet).save(os.path.join(OUT, "contact_sheet.png"))
        print("saved contact_sheet.png", sheet.shape)

    # 2) grasp GIFs — diverse objects/sequences
    seen_obj = set(); ng = 0
    for nc, head, seq, side, hk, ts, obj in picks:
        if obj in seen_obj: continue
        seen_obj.add(obj)
        seq_dir = os.path.join(DS[head], seq)
        d = np.load(os.path.join(DATA, head, f"{seq}.npz"), allow_pickle=True)
        outp = os.path.join(OUT, "gifs", f"{obj}_{seq}_{side}.gif")
        try:
            r = VZ.make_grasp_gif(mano, lib, seq_dir, d, hk, outp)
            if r: index["gifs"].append(os.path.basename(outp)); ng += 1; print("gif", os.path.basename(outp))
        except Exception as e:
            print("  gif fail", seq, e)
        if ng >= 6: break

    json.dump(index, open(os.path.join(OUT, "gallery_index.json"), "w"), indent=1)
    print("DONE gallery: frames", len(index["frames"]), "gifs", len(index["gifs"]))

if __name__ == "__main__":
    main()
