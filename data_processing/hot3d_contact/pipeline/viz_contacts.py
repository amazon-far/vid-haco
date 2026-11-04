#!/usr/bin/env python3
"""Visualize HOT3D dense hand-contact maps.

Renders (offscreen, EGL):
  - per-frame 2-panel: [hand colored by contact map] | [hand + object(s) in context]
  - contact-evolution GIFs over a grasp
  - dataset-wide per-vertex contact-frequency hand heatmap (HACO 'contact_means')
Hand geometry recomputed from HOT3D MANO annotations; contact from the npz.
"""
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys, json, glob
import numpy as np
import trimesh
import pyrender
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hot3d_contact_lib as L

SKIN = np.array([236, 199, 164], np.uint8)
RED = np.array([214, 40, 40], np.uint8)
OBJ = np.array([170, 176, 184], np.uint8)
W, H = 480, 480

def _cam_pose(center, radius, az=35, el=22, dist_mul=2.6):
    az, el = np.radians(az), np.radians(el)
    d = radius * dist_mul
    eye = center + d * np.array([np.cos(el)*np.sin(az), np.sin(el), np.cos(el)*np.cos(az)])
    f = center - eye; f /= np.linalg.norm(f)
    up = np.array([0, 1.0, 0])
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4); M[:3, 0] = s; M[:3, 1] = u; M[:3, 2] = -f; M[:3, 3] = eye
    return M

def _scene(meshes, center, radius, az=35, el=22):
    sc = pyrender.Scene(bg_color=[255, 255, 255, 0], ambient_light=[0.35, 0.35, 0.35])
    for m in meshes:
        sc.add(m)
    cam = pyrender.PerspectiveCamera(yfov=np.pi/4.5)
    cp = _cam_pose(center, radius, az, el)
    sc.add(cam, pose=cp)
    for a in (az-40, az+50):
        lp = _cam_pose(center, radius, a, el+30)
        sc.add(pyrender.DirectionalLight(color=[1,1,1], intensity=3.2), pose=lp)
    return sc

def _hand_trimesh(verts, faces, contact=None, dist=None):
    m = trimesh.Trimesh(verts, faces, process=False)
    vc = np.tile(np.append(SKIN, 255), (len(verts), 1)).astype(np.uint8)
    if contact is not None:
        cc = contact > 0
        vc[cc, :3] = RED
    m.visual.vertex_colors = vc
    return m

_R = {}
def _renderer():
    if "r" not in _R:
        _R["r"] = pyrender.OffscreenRenderer(W, H)
    return _R["r"]

def render_panel(hand_verts, hand_faces, contact, obj_world, az=35, el=22, with_obj=True):
    """obj_world: list of (verts, faces). Returns HxWx3 uint8."""
    meshes = []
    hm = _hand_trimesh(hand_verts, hand_faces, contact)
    meshes.append(pyrender.Mesh.from_trimesh(hm, smooth=False))
    allv = [hand_verts]
    if with_obj:
        for v, f in obj_world:
            om = trimesh.Trimesh(v, f, process=False)
            om.visual.vertex_colors = np.append(OBJ, 235)
            meshes.append(pyrender.Mesh.from_trimesh(om, smooth=True))
            allv.append(v)
    allv = np.vstack(allv)
    center = 0.5*(allv.min(0)+allv.max(0)); radius = np.linalg.norm(allv.max(0)-allv.min(0))/2
    sc = _scene(meshes, center, radius, az, el)
    color, _ = _renderer().render(sc, flags=pyrender.RenderFlags.RGBA)
    rgb = color[..., :3].copy()
    a = color[..., 3]
    rgb[a == 0] = 255
    return rgb

def _label(img, text, sub=""):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    try:
        fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        fnt = fs = ImageFont.load_default()
    d.rectangle([0, 0, im.width, 22], fill=(30, 30, 40))
    d.text((5, 3), text, fill=(255, 255, 255), font=fnt)
    if sub:
        d.text((5, im.height-18), sub, fill=(20, 20, 20), font=fs)
    return np.array(im)

def frame_geometry(mano, lib, hands_t, objs_t, hk):
    e = hands_t[hk]
    v = mano.forward(hk, [e['betas']], [e['pose']], [e['q_wxyz']], [e['t']], batch=False).numpy()
    faces = mano.faces[hk]
    ow = []
    for uid, M in objs_t:
        vv, ff = lib.mesh(uid).vertices, lib.mesh(uid).faces
        ow.append((np.asarray(vv) @ M[:3,:3].T + M[:3,3], np.asarray(ff)))
    return v, faces, ow

def _cmap(vals):
    """white -> yellow -> red -> dark-red colormap for freq in [0,1]-ish (auto-normed)."""
    v = np.clip(vals / max(1e-6, vals.max()), 0, 1)
    stops = np.array([[245,245,245],[255,225,120],[240,120,40],[150,10,10]], float)
    pos = np.array([0,0.33,0.66,1.0])
    out = np.zeros((len(v),3))
    for c in range(3):
        out[:,c] = np.interp(v, pos, stops[:,c])
    return out.astype(np.uint8)

def render_template_heatmap(mano, means, out_path, hk=None):
    """Render MANO template hand colored by per-vertex contact frequency, 2 views."""
    hk = hk or L.RIGHT
    import torch
    v = mano.layer[hk](torch.zeros(1,10), torch.zeros(1,3), torch.zeros(1,15),
                       torch.zeros(1,3))[0].detach().numpy()
    faces = mano.faces[hk]
    cols = _cmap(means)
    m = trimesh.Trimesh(v, faces, process=False)
    m.visual.vertex_colors = np.concatenate([cols, np.full((len(v),1),255,np.uint8)],1)
    center = 0.5*(v.min(0)+v.max(0)); radius = np.linalg.norm(v.max(0)-v.min(0))/2
    imgs=[]
    for az,lbl in ((0,"palm"),(180,"back")):
        sc=_scene([pyrender.Mesh.from_trimesh(m,smooth=False)],center,radius,az=az,el=8)
        color,_=_renderer().render(sc,flags=pyrender.RenderFlags.RGBA)
        rgb=color[...,:3].copy(); rgb[color[...,3]==0]=255
        imgs.append(_label(rgb,f"contact frequency ({lbl})",f"peak {100*means.max():.0f}%  mean {100*means.mean():.1f}%"))
    Image.fromarray(np.concatenate(imgs,1)).save(out_path)
    return out_path

def render_single(mano, lib, seq_dir_hands, seq_dir_objs, npz, ts, hk, az=40, el=22, with_obj=True):
    side = "left" if hk == L.LEFT else "right"
    i = int(np.where(npz["timestamps_ns"] == ts)[0][0])
    contact = npz[f"{side}_contact"][i]
    v, faces, ow = frame_geometry(mano, lib, seq_dir_hands[ts], seq_dir_objs.get(ts, []), hk)
    return render_panel(v, faces, contact, ow, az=az, el=el, with_obj=with_obj), int((contact>0).sum())

def make_grasp_gif(mano, lib, seq_dir, npz, hk, out_path, n=48, stride=2):
    """GIF of contact evolving through the window of strongest contact."""
    import imageio
    hands = L.load_hand_traj(seq_dir); objs = L.load_object_traj(seq_dir)
    side = "left" if hk == L.LEFT else "right"
    C = npz[f"{side}_contact"]; ts = npz["timestamps_ns"]
    nc = C.sum(1); peak = int(nc.argmax())
    lo = max(0, peak - n*stride//2); hi = min(len(ts), lo + n*stride)
    frames=[]
    for i in range(lo, hi, stride):
        t=int(ts[i])
        if t not in hands or hk not in hands[t]: continue
        v,faces,ow = frame_geometry(mano,lib,hands[t],objs.get(t,[]),hk)
        img = render_panel(v,faces,C[i],ow,az=40,el=22,with_obj=True)
        img = _label(img,f"{os.path.basename(seq_dir)} {side}",f"{int(C[i].sum())}/778 contact")
        frames.append(img)
    if frames:
        imageio.mimsave(out_path, frames, duration=0.08, loop=0)
    return out_path if frames else None

def render_frame_2panel(mano, lib, seq_dir, npz, ts, hk):
    hands = L.load_hand_traj(seq_dir); objs = L.load_object_traj(seq_dir)
    side = "left" if hk == L.LEFT else "right"
    i = int(np.where(npz["timestamps_ns"] == ts)[0][0])
    contact = npz[f"{side}_contact"][i]
    v, faces, ow = frame_geometry(mano, lib, hands[ts], objs.get(ts, []), hk)
    # object(s) actually touched
    near = npz[f"{side}_nearest_obj"][i]
    touched = [npz["object_names"][g] for g in np.unique(near[near >= 0])]
    pA = render_panel(v, faces, contact, ow, az=15, el=18, with_obj=False)
    pB = render_panel(v, faces, contact, ow, az=40, el=22, with_obj=True)
    nc = int((contact > 0).sum())
    pA = _label(pA, f"{side} hand contact map", f"{nc}/778 verts in contact @1cm")
    pB = _label(pB, "hand + object(s)", f"touching: {', '.join(map(str,touched)) or 'none'}")
    return np.concatenate([pA, pB], axis=1)
