"""
Mesh-top contact visualization from a cop-viz COMBINED npz.

Reads ONE combined npz (the same file cop-viz's make_clips / cop_fit / bake stages consume) instead of
the separate Stage-A/B/tracking npz files. Everything the render needs is already in the combined npz:
    frame_idx, fps, bbox_sm            (Stage A meta)
    contact_prob                       (Stage B contact map, [N,778])
    verts, cam_t, focal, found, faces  (hand tracking, HaWoR)
    hand_side, img_wh                  (self-describing; optional)

Layout (1000x1500):
    +-----------------------------------------------+-----------------+
    | TOP  full frame + bbox + banner + hand mesh   | posed hand      |
    |      (semi-transparent overlay, in place)     | PALMAR (+contact)|
    +-----------------------+-----------------------+-----------------+
    | Contact: Dorsal       | Contact: Palmar       | posed hand      |
    |   (rest-pose hand)    |   (rest-pose hand)    | DORSAL (+contact)|
    +-----------------------+-----------------------+-----------------+

⚠️ RUNTIME: this needs the `haco_render` conda env (pyrender/EGL), this project's `lib` on PYTHONPATH,
and data/base_data/mano_assets.npz (built by tools/make_mano_assets.py from your MANO_RIGHT.pkl).
`run.py` sets all of that up.

The `--combined` npz carries ABSOLUTE frame ids, so a sub-range renders directly against the full
source video (frames are indexed by frame_idx).

SHARDING (this project's addition to the original single-process renderer): `--start_idx/--end_idx`
render one contiguous slice of rows, so N shards can run on N GPUs and be concatenated with
`ffmpeg -f concat -c copy` — every shard writes identical libx264/yuv420p/crf20 at the same size
and fps. `--warmup` runs the causal contact-mask EMA over rows before the window and discards them,
so the mask is already converged at the shard's first rendered frame and seams are invisible.
"""
import os, argparse, subprocess
from fractions import Fraction
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")   # before PyOpenGL is imported (by pyrender)
import numpy as np, cv2, trimesh, pyrender

from lib.utils import mano_assets
from lib.utils.contact_mask import ema, drop_small_components
from lib.utils.imgseq_capture import open_capture   # image-seq dir -> imread-faithful capture (mp4 -> cv2)

CELL = 500
ROT180X = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
DEFAULT_COL = np.array([130, 130, 130, 255], np.uint8)   # gray (no contact)
CONTACT_COL = np.array([0, 255, 0, 255], np.uint8)       # green (contact)
MCP = [1, 4, 10, 7]   # index, middle, ring, pinky MCP joints (orig MANO order)
POSED_HALF = 0.155    # m, half the panel width for a posed hand (an open hand's fingertips stay inside)
REST_HALF = 0.12      # m, the same for the flat rest-pose hand, which rest_hand_view centres


def fit_rect(img_bgr, cw, ch, bg=(18, 18, 18)):
    h, w = img_bgr.shape[:2]
    s = min(cw / w, ch / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    r = cv2.resize(img_bgr, (nw, nh))
    out = np.full((ch, cw, 3), bg, np.uint8)
    y0, x0 = (ch - nh) // 2, (cw - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = r
    return out


def label(img, text, org=(12, 30), color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return img


def wilor_overlay_full(rndr, verts, faces, cam_t, focal, W, H, rgb, alpha=0.6):
    mesh = trimesh.Trimesh(verts + cam_t[None], faces, process=False)
    mesh.apply_transform(ROT180X)
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=(0.4, 0.4, 0.4))
    mat = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, roughnessFactor=0.7,
                                             alphaMode="OPAQUE", baseColorFactor=(0.62, 0.66, 0.80, 1.0))
    scene.add(pyrender.Mesh.from_trimesh(mesh, material=mat))
    scene.add(pyrender.IntrinsicsCamera(fx=focal, fy=focal, cx=W / 2.0, cy=H / 2.0, zfar=1e4), pose=np.eye(4))
    dl = pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0)
    for ang in (np.radians(30), np.radians(-30), 0.0):
        p = trimesh.transformations.rotation_matrix(ang, [0, 1, 0]); scene.add(dl, pose=p)
    color, _ = rndr.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0
    a = color[:, :, 3:4] * float(alpha)
    comp = color[:, :, :3] * a + (rgb.astype(np.float32) / 255.0) * (1 - a)
    return (comp * 255).astype(np.uint8)


def top_banner(cell, n_contact, hand_label='Right'):
    on = int(n_contact) > 0
    txt = f"{hand_label} hand contact: {'YES' if on else 'NO'} ({int(n_contact)} verts)"
    col = (40, 200, 40) if on else (170, 170, 170)
    ov = cell.copy(); cv2.rectangle(ov, (0, 0), (cell.shape[1], 46), (0, 0, 0), -1)
    cell = cv2.addWeighted(ov, 0.55, cell, 0.45, 0)
    cv2.putText(cell, txt, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(cell, txt, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
    label(cell, "Input + bbox + 3D hand mesh", org=(12, cell.shape[0] - 16))
    return cell


def wrist_fix(verts, J_regressor):
    """Re-express posed verts in a canonical wrist frame (global orientation removed): fingers up, back of
    the hand towards +z. A proper rotation, so a right hand still looks like a right hand."""
    J = J_regressor @ verts                          # [16,3]
    wrist = J[0]; mcp = J[MCP]
    up = mcp.mean(0) - wrist; up /= (np.linalg.norm(up) + 1e-8)
    across_raw = J[1] - J[7]
    normal = np.cross(up, across_raw); normal /= (np.linalg.norm(normal) + 1e-8)
    across = np.cross(up, normal)
    R = np.stack([across, up, normal], 0)
    center = 0.5 * (wrist + mcp.mean(0))
    return (verts - center) @ R.T


def render_hand(rndr, verts, vcolors, faces, ry_deg, half_extent):
    """CELL x CELL BGR panel of a hand mesh with per-vertex colours, on white.

    `verts` are in a fingers-up frame centred on the panel (wrist_fix / rest_hand_view); the hand is first
    turned `ry_deg` about the vertical axis (0 = back of the hand towards the viewer, 180 = palm). The view
    is orthographic, `half_extent` metres either side of the centre, so every frame is drawn at one scale;
    a key light shines from the viewer and two fill lights from above left and above right.
    """
    m = trimesh.Trimesh(verts, faces, process=False)
    m.visual.vertex_colors = vcolors
    if ry_deg:
        m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(ry_deg), [0, 1, 0]))
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=(0.35, 0.35, 0.35))
    scene.add(pyrender.Mesh.from_trimesh(m, smooth=False))
    view = np.eye(4); view[2, 3] = 1.0                 # 1 m in front of the hand, looking down -z
    scene.add(pyrender.OrthographicCamera(xmag=half_extent, ymag=half_extent, znear=0.05, zfar=2.0), pose=view)
    for tilt, turn, power in ((0.0, 0.0, 1.6), (-35.0, 30.0, 0.9), (-35.0, -30.0, 0.9)):
        pose = trimesh.transformations.euler_matrix(np.radians(tilt), np.radians(turn), 0.0, 'sxyz')
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=power), pose=pose)
    rgb, _ = rndr.render(scene)
    return rgb[:, :, ::-1].copy()


def rest_hand_view(v_template, J_regressor):
    """The rest-pose (flat) hand, fingers up, back of the hand towards the camera (thumb on the left,
    as on a real right hand; turn it 180 degrees about y for the palm), centred in the panel."""
    J = J_regressor @ v_template
    wrist, mcp = J[0], J[MCP]
    up = mcp.mean(0) - wrist; up /= np.linalg.norm(up)
    toward_index = J[1] - J[7]                      # pinky MCP -> index MCP
    back = np.cross(up, toward_index); back /= np.linalg.norm(back)
    side = np.cross(up, back)                       # completes a proper (det +1) rotation
    P = (v_template - wrist) @ np.stack([side, up, back], 0).T
    P[:, :2] -= 0.5 * (P[:, :2].min(0) + P[:, :2].max(0))   # the flat hand is longer than a posed one
    return P


def h264_pipe(path, fps, width, height):
    """An ffmpeg process that encodes raw BGR frames from stdin to `path`. Every shard uses exactly these
    settings, which is what lets run.py join shards with a stream copy."""
    rate = Fraction(float(fps)).limit_denominator(1001)   # e.g. 29.97002997 -> 30000/1001
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
           '-s', f'{width}x{height}', '-framerate', f'{rate.numerator}/{rate.denominator}', '-i', '-',
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def blank_cell(text=''):
    img = np.full((CELL, CELL, 3), 245, np.uint8)
    if text:
        cv2.putText(img, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description="Mesh-top contact viz from a cop-viz COMBINED npz.")
    ap.add_argument('--combined', required=True, help='cop-viz combined npz (frame_idx/bbox_sm/contact_prob/verts/cam_t/focal/found/faces)')
    ap.add_argument('--video', required=True, help='source mp4 (indexed by ABSOLUTE frame_idx, so clips work)')
    ap.add_argument('--out_h264', required=True, help='output H.264 mp4')
    ap.add_argument('--sample_dir', default='', help='dir for a few sample PNG frames (default: <out_h264>_samples)')
    ap.add_argument('--thr', type=float, default=-1.0, help='contact threshold (default: read from npz, else 0.52)')
    ap.add_argument('--mesh_alpha', type=float, default=0.6, help='overlay mesh opacity (top panel)')
    ap.add_argument('--max_frames', type=int, default=0, help='>0: only render the first N (debug)')
    ap.add_argument('--hand', default='', choices=['', 'right', 'left'], help='banner label (default: npz hand_side, else right)')
    # --- sharding: render one contiguous slice of the combined npz, for concat with its siblings ---
    ap.add_argument('--start_idx', type=int, default=0, help='first ROW of the combined npz to render (not a frame id)')
    ap.add_argument('--end_idx', type=int, default=-1, help='one past the last row to render; -1 = to the end')
    ap.add_argument('--warmup', type=int, default=48,
                    help='rows of contact-mask EMA to run before --start_idx so a shard boundary is invisible')
    args = ap.parse_args()

    out_h264 = args.out_h264
    args.sample_dir = args.sample_dir or (os.path.splitext(out_h264)[0] + '_samples')
    for d in (os.path.dirname(out_h264), args.sample_dir):
        if d:
            os.makedirs(d, exist_ok=True)

    assets = mano_assets.load()
    J_reg = assets['J_regressor'].astype(np.float64)
    wt_faces = assets['watertight_faces']
    n_verts = mano_assets.NUM_VERTS

    # ---- read EVERYTHING from the one combined npz ----
    Z = np.load(args.combined, allow_pickle=True)
    need_keys = ['frame_idx', 'fps', 'bbox_sm', 'contact_prob', 'verts', 'cam_t', 'focal', 'found', 'faces']
    missing = [k for k in need_keys if k not in Z.files]
    assert not missing, f"combined npz missing keys: {missing} (have {sorted(Z.files)})"
    fps = float(Z['fps']); a_idx = Z['frame_idx']; bbox_sm = Z['bbox_sm']
    contact_prob = Z['contact_prob']
    verts, cam_t, focal, found = Z['verts'], Z['cam_t'], Z['focal'], Z['found']
    faces = Z['faces']
    thr = float(args.thr) if args.thr >= 0 else (float(Z['thr']) if 'thr' in Z.files else 0.52)
    hand = args.hand or (str(Z['hand_side']) if 'hand_side' in Z.files else 'right')
    n_all = len(a_idx) if args.max_frames <= 0 else min(args.max_frames, len(a_idx))

    # ---- window this shard renders -------------------------------------------------------
    # Rows, not frame ids: a shard is a contiguous slice of the combined npz. PASS 0 below is
    # a causal EMA over contact masks, so a shard that simply started at its own first row
    # would begin with an un-converged mask and show a visible pop at the seam. We therefore
    # run the EMA from `warmup` rows earlier and discard those rows. At alpha=0.8 the weight
    # of a sample k rows back is 0.8^k, so 48 rows leaves ~2e-5 of the pre-window state —
    # far below the >0.5 mask threshold, making shards agree with a single-process render.
    i0 = max(0, min(args.start_idx, n_all))
    i1 = n_all if args.end_idx < 0 else max(i0, min(args.end_idx, n_all))
    warm = max(0, i0 - max(0, args.warmup))
    N = i1 - i0
    if N <= 0:
        raise SystemExit(f"[meshtop] empty window rows [{i0},{i1}) of {n_all}")
    pose_valid = np.abs(verts).reshape(len(verts), -1).sum(1) > 1e-6
    sharded = (i0, i1) != (0, n_all)
    print(f"[meshtop] combined={os.path.basename(args.combined)} rows[{i0},{i1}) of {n_all} N={N} "
          f"fps={fps:.3f} hand={hand} thr={thr:.2f} abs[{int(a_idx[i0])}..{int(a_idx[i1-1])}] "
          f"wilor_found={int(found[i0:i1].sum())}/{N} pose_valid={int(pose_valid[i0:i1].sum())}"
          + (f" warmup_rows={i0 - warm}" if sharded else ""), flush=True)

    # decode the frames we need (indexed by ABSOLUTE frame_idx -> clips render vs the full video)
    need = set(int(a_idx[k]) for k in range(i0, i1)); full = {}
    cap = open_capture(args.video); fi = 0; mx = max(need)
    while True:
        ok, fr = cap.read()
        if not ok or fi > mx:
            break
        if fi in need:
            full[fi] = fr
        fi += 1
    cap.release()
    assert full, "no frames decoded — check --video matches the combined npz's frame_idx space"
    H, W = next(iter(full.values())).shape[:2]
    from tqdm import tqdm

    # ---- PASS 0: causal contact-mask smoothing, shared by all panels ----
    # Runs from `warm` so the EMA is converged by i0; only rows in the window are kept.
    cmask_all = np.zeros((N, n_verts), bool); smoothed = None
    for k in range(warm, i1):
        smoothed = ema(smoothed, contact_prob[k] >= thr, alpha=0.8)
        if k < i0:
            continue                      # warm-up row: advance the EMA, render nothing
        cmask_all[k - i0] = drop_small_components(smoothed > 0.5, wt_faces, min_size=20)

    # From here on everything is window-local: row k of these arrays is row i0+k of the npz.
    a_idx = a_idx[i0:i1]
    bbox_sm = bbox_sm[i0:i1]
    verts, cam_t, focal, found = verts[i0:i1], cam_t[i0:i1], focal[i0:i1], found[i0:i1]
    pose_valid = pose_valid[i0:i1]

    # ---- PASS 1: tracked hand mesh overlaid on the FULL frame + bbox, fit to the wide top cell ----
    top_cells = np.zeros((N, CELL, 2 * CELL, 3), np.uint8)
    overlay_rndr = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H, point_size=1.0)
    for k in tqdm(range(N), desc='[meshtop] pass1 overlay'):
        fbgr = full[int(a_idx[k])]
        if bool(found[k]):
            rgb = cv2.cvtColor(fbgr, cv2.COLOR_BGR2RGB)
            ov = wilor_overlay_full(overlay_rndr, verts[k], faces, cam_t[k], float(focal[k]), W, H, rgb, args.mesh_alpha)
            ovb = cv2.cvtColor(ov, cv2.COLOR_RGB2BGR)
        else:
            ovb = fbgr.copy()
        x, y, w, h = [float(v) for v in bbox_sm[k]]
        cv2.rectangle(ovb, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 3)
        top_cells[k] = fit_rect(ovb, 2 * CELL, CELL)
    try:
        overlay_rndr.delete()
    except Exception:
        pass

    # ---- PASS 2: tracked, wrist-fixed hand painted with the contact map; palmar + dorsal ----
    right_cells = np.zeros((N, 2 * CELL, CELL, 3), np.uint8)
    posed_rndr = pyrender.OffscreenRenderer(viewport_width=CELL, viewport_height=CELL, point_size=1.0)
    for k in tqdm(range(N), desc='[meshtop] pass2 posed'):
        if pose_valid[k]:
            vc = wrist_fix(verts[k], J_reg)
            vcol = np.tile(DEFAULT_COL, (n_verts, 1)); vcol[cmask_all[k]] = CONTACT_COL
            palmar = label(render_hand(posed_rndr, vc, vcol, faces, 180.0, POSED_HALF), "Posed hand: Palmar")
            dorsal = label(render_hand(posed_rndr, vc, vcol, faces, 0.0, POSED_HALF), "Posed hand: Dorsal")
        else:
            palmar = label(blank_cell("Hand pose: n/a"), "Posed hand: Palmar")
            dorsal = blank_cell()
        right_cells[k] = np.vstack([palmar, dorsal])

    # ---- PASS 3: rest-pose contact panels (same CELL renderer) + banner + assemble + write ----
    rest = rest_hand_view(assets['v_template'].astype(np.float64), J_reg)
    writer = None
    order = np.argsort(-cmask_all[:N].sum(1))
    sample_pos = sorted({int(order[0]), int(order[len(order) // 2]), int(N // 8)})
    for k in tqdm(range(N), desc='[meshtop] pass3 contact'):
        fbgr = full[int(a_idx[k])]
        cmask = cmask_all[k]
        top = top_banner(top_cells[k].copy(), int(cmask.sum()), hand_label=hand.capitalize())
        vcol = np.tile(DEFAULT_COL, (n_verts, 1)); vcol[cmask] = CONTACT_COL
        dorsal = label(render_hand(posed_rndr, rest, vcol, wt_faces, 0.0, REST_HALF), "Contact: Dorsal")
        palmar = label(render_hand(posed_rndr, rest, vcol, wt_faces, 180.0, REST_HALF), "Contact: Palmar")

        left = np.vstack([top, np.hstack([dorsal, palmar])])     # 1000x1000
        grid = np.hstack([left, right_cells[k]])                 # 1000x1500
        cv2.line(grid, (0, CELL), (2 * CELL, CELL), (60, 60, 60), 2)
        cv2.line(grid, (CELL, CELL), (CELL, 2 * CELL), (60, 60, 60), 2)
        cv2.line(grid, (2 * CELL, 0), (2 * CELL, 2 * CELL), (60, 60, 60), 2)
        cv2.line(grid, (2 * CELL, CELL), (3 * CELL, CELL), (60, 60, 60), 2)

        if writer is None:
            writer = h264_pipe(out_h264, fps, grid.shape[1], grid.shape[0])
            print(f"[meshtop] grid = {grid.shape[1]}x{grid.shape[0]}", flush=True)
        writer.stdin.write(np.ascontiguousarray(grid).tobytes())
        if k in sample_pos:
            p = os.path.join(args.sample_dir, f'frame_{int(a_idx[k]):05d}.png'); cv2.imwrite(p, grid)
            print(f"[meshtop] sample -> {p}", flush=True)

    try:
        posed_rndr.delete()
    except Exception:
        pass
    writer.stdin.close()
    if writer.wait() != 0:
        raise SystemExit(f"[meshtop] ffmpeg failed writing {out_h264}")
    print(f"[meshtop] wrote {out_h264}", flush=True)


if __name__ == '__main__':
    main()
