"""DECISIVE CHECK: derive bbox_hand_r for the allocentric views and draw it on the real images.

Derivation (ARCTIC's crop geometry, crop_coords.py):
  j2d = data_dict[seq]['2d']['joints.right'][row, view]      # 21 pts, ORIGINAL image coords
  cropped = to_crop_coords(j2d, data_dict[seq]['bbox'][row, view])   # -> 1000x1000 cropped coords
  bbox_hand_r = tight box of those 21 pts, expanded 2.0x   (== HACO's cam-0 scale, w=2.015 h=1.977)

Matching HACO's cam-0 numbers bit-for-bit is NOT the goal: 23% of HACO's ARCTIC boxes are degenerate
(center off-image; widths to 70,746 px). The goal is a box that lands on the hand. So: draw it.
"""
import numpy as np
import cv2, os, glob

from crop_coords import to_crop_coords

SEQ = 's01/box_grab_01'
IMROOT = 'data/ARCTIC/data/cropped_images'
OUT = 'data/arctic_allo_work/_viz'
os.makedirs(OUT, exist_ok=True)
EXPAND = 2.0
OFF = 10          # frame 10 -> row 0


def bbox_from(k, expand=EXPAND):
    x0, y0, x1, y1 = k[:, 0].min(), k[:, 1].min(), k[:, 0].max(), k[:, 1].max()
    cx, cy, w, h = (x0+x1)/2, (y0+y1)/2, (x1-x0), (y1-y0)
    w *= expand; h *= expand
    return np.array([cx - w/2, cy - h/2, w, h])


print("loading p1_train.npy ...", flush=True)
p1 = np.load('data/arctic_allo_work/p1_train.npy', allow_pickle=True).item()
e = p1['data_dict'][SEQ]
j2d = e['2d']['joints.right']       # (F,10,21,2)
bb = e['bbox']                      # (F,9,3)

frames = [200, 400, 600]
views = [1, 3, 5, 8]
tiles = []
for n in frames:
    r = n - OFF
    row = []
    for v in views:
        p = os.path.join(IMROOT, SEQ, str(v), f'{n:05d}.jpg')
        im = cv2.imread(p)
        if im is None:
            print(f"  MISSING {p}"); continue
        k = to_crop_coords(j2d[r, v], bb[r, v])
        b = bbox_from(k)
        vis = im.copy()
        # draw the 21 joints + the derived box
        for pt in k.astype(int):
            cv2.circle(vis, tuple(pt), 5, (60, 255, 60), -1)
        cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[0]+b[2]), int(b[1]+b[3])), (0, 200, 255), 5)
        cv2.rectangle(vis, (0, 0), (1000, 46), (0, 0, 0), -1)
        cv2.putText(vis, f'cam {v}  frame {n}', (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
        row.append(cv2.resize(vis, (330, 330)))
    if row:
        tiles.append(np.hstack(row))
cv2.imwrite(f'{OUT}/allo_bbox_check.jpg', np.vstack(tiles))
print(f"wrote {OUT}/allo_bbox_check.jpg")

# also: roughly the 256x256 hand crop the loader produces
crops = []
for n in frames:
    r = n - OFF
    row = []
    for v in views:
        im = cv2.imread(os.path.join(IMROOT, SEQ, str(v), f'{n:05d}.jpg'))
        if im is None:
            continue
        k = to_crop_coords(j2d[r, v], bb[r, v])
        b = bbox_from(k)
        cx, cy = b[0]+b[2]/2, b[1]+b[3]/2
        s = max(b[2], b[3])
        x0, y0 = int(cx - s/2), int(cy - s/2)
        H, W = im.shape[:2]
        pad = 400
        imp = cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        c = imp[y0+pad:y0+pad+int(s), x0+pad:x0+pad+int(s)]
        if c.size == 0:
            continue
        c = cv2.resize(c, (256, 256))
        cv2.putText(c, f'cam{v} f{n}', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        row.append(c)
    if row:
        crops.append(np.hstack(row))
cv2.imwrite(f'{OUT}/allo_crops.jpg', np.vstack(crops))
print(f"wrote {OUT}/allo_crops.jpg")
