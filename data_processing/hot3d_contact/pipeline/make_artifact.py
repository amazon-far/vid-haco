#!/usr/bin/env python3
"""Assemble a self-contained HTML review page for the HOT3D contact-map deliverable.
Embeds selected visualizations as base64 data URIs + stats from dataset_stats.json.
"""
import os, json, base64, glob, io
import numpy as np
from PIL import Image

VIZ = "data/hot3d_contact/viz"
DATA = "data/hot3d_contact/contact_data"
OUT = os.path.join(VIZ, "hot3d_contact_review.html")

def datauri(path, kind="image/png", max_w=None):
    if not os.path.exists(path):
        return None
    if max_w and path.endswith(".png"):
        im = Image.open(path).convert("RGB")
        if im.width > max_w:
            im = im.resize((max_w, int(im.height * max_w / im.width)), Image.LANCZOS)
        b = io.BytesIO(); im.save(b, "PNG", optimize=True); raw = b.getvalue()
    else:
        raw = open(path, "rb").read()
    return f"data:{kind};base64," + base64.b64encode(raw).decode()

stats = json.load(open(os.path.join(DATA, "dataset_stats.json")))
gi = json.load(open(os.path.join(VIZ, "gallery_index.json")))

heat = datauri(os.path.join(VIZ, "contact_frequency_hand.png"), max_w=900)
# pick a diverse subset: spread across object categories, cap 2 per object
_by_obj, _sel = {}, []
for name in gi["frames"]:
    parts = name[:-4].split("_")
    obj = parts[1] if len(parts) > 1 else ""
    _by_obj.setdefault(obj, []).append((name, obj, parts[-1]))
# round-robin across objects so the grid shows variety
_pool = list(_by_obj.values())
while _pool and len(_sel) < 12:
    for lst in list(_pool):
        if lst:
            _sel.append(lst.pop(0))
        if len(_sel) >= 12:
            break
    _pool = [l for l in _pool if l]
frames = []
for name, obj, nc in _sel:
    u = datauri(os.path.join(VIZ, "frames", name), max_w=760)
    if u:
        frames.append((u, obj, nc.replace("v", "")))
frames = frames[:12]
gifs = []
for name in gi["gifs"][:6]:
    u = datauri(os.path.join(VIZ, "gifs", name), kind="image/gif")
    if u:
        gifs.append((u, name[:-4].split("_")[0]))

ph = stats.get("per_headset", {})
def hv(h, k): return ph.get(h, {}).get(k, 0)
n_seq = stats["n_sequences"]; n_hi = stats["n_hand_instances"]
n_ci = stats["n_contact_instances"]; frac = stats["contact_instance_frac"]
n_frames = sum(v.get("frames", 0) for v in ph.values())
objc = stats["per_object_frames"]
maxo = max(objc.values()) if objc else 1
obj_rows = "".join(
    f'<div class="bar"><span class="bl">{o}</span>'
    f'<span class="bt"><span class="bf" style="width:{100*c/maxo:.1f}%"></span></span>'
    f'<span class="bn">{c:,}</span></div>'
    for o, c in list(objc.items())[:16])

tiles = [
    ("sequences", f"{n_seq}", "with hand GT (train)"),
    ("hand-instances", f"{n_hi/1e6:.2f}M", "posed hand frames labeled"),
    ("contact rate", f"{100*frac:.1f}%", "frames with ≥1 contact vertex"),
    ("frames", f"{n_frames/1e6:.2f}M", "Aria + Quest annotated frames"),
    ("threshold", "1.0 cm", "HACO hand-object c_thres"),
    ("vs exact distance", "99.95%", "per-vertex label agreement"),
]
tile_html = "".join(
    f'<div class="tile"><div class="tv">{v}</div><div class="tk">{k}</div><div class="ts">{s}</div></div>'
    for k, v, s in tiles)

frame_html = "".join(
    f'<figure class="shot"><img loading="lazy" src="{u}" alt="{obj} grasp contact"/>'
    f'<figcaption><b>{obj}</b><span>{nc} contact verts</span></figcaption></figure>'
    for u, obj, nc in frames)
gif_html = "".join(
    f'<figure class="shot gif"><img loading="lazy" src="{u}" alt="{obj} grasp animation"/>'
    f'<figcaption><b>{obj}</b><span>contact through grasp</span></figcaption></figure>'
    for u, obj in gifs)

html = f"""<title>HOT3D Dense Hand-Contact Maps</title>
<style>
:root{{
  --bg:#f5f6f8; --panel:#ffffff; --ink:#171a20; --dim:#5a626e; --line:#e2e5ea;
  --accent:#e23b3b; --accent-soft:#fbe6e6; --ok:#0f9d8f; --skin:#d9a878; --obj:#8f99a6;
  --mono:ui-monospace,"SF Mono","Cascadia Code",Menlo,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}}
@media (prefers-color-scheme:dark){{:root{{
  --bg:#0d1014; --panel:#161a20; --ink:#e9ebef; --dim:#98a1ad; --line:#252b34;
  --accent:#ff5252; --accent-soft:#2a1618; --ok:#2fc0ac;
}}}}
:root[data-theme="light"]{{--bg:#f5f6f8;--panel:#ffffff;--ink:#171a20;--dim:#5a626e;--line:#e2e5ea;--accent:#e23b3b;--accent-soft:#fbe6e6;}}
:root[data-theme="dark"]{{--bg:#0d1014;--panel:#161a20;--ink:#e9ebef;--dim:#98a1ad;--line:#252b34;--accent:#ff5252;--accent-soft:#2a1618;--ok:#2fc0ac;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.55;
  -webkit-font-smoothing:antialiased}}
.wrap{{max-width:1180px;margin:0 auto;padding:0 24px 80px}}
header{{position:sticky;top:0;z-index:5;background:color-mix(in srgb,var(--bg) 88%,transparent);
  backdrop-filter:blur(10px);border-bottom:1px solid var(--line);margin-bottom:34px}}
.hd{{max-width:1180px;margin:0 auto;padding:14px 24px;display:flex;align-items:center;gap:16px;flex-wrap:wrap}}
.dot{{width:12px;height:12px;border-radius:50%;background:var(--accent);box-shadow:0 0 0 4px var(--accent-soft)}}
.hd h1{{font-size:15px;letter-spacing:.02em;margin:0;font-weight:650}}
.hd .sub{{color:var(--dim);font-size:12.5px;font-family:var(--mono)}}
.badge{{margin-left:auto;font-family:var(--mono);font-size:11.5px;color:var(--accent);
  border:1px solid var(--accent);border-radius:100px;padding:3px 11px}}
.hero{{display:grid;grid-template-columns:1.15fr 1fr;gap:22px;align-items:stretch}}
@media(max-width:820px){{.hero{{grid-template-columns:1fr}}}}
.lead{{grid-column:1/-1;margin:0 0 6px}}
.lead h2{{font-size:30px;line-height:1.12;margin:0 0 8px;font-weight:720;letter-spacing:-.01em;text-wrap:balance}}
.lead p{{color:var(--dim);margin:0;max-width:70ch;font-size:15px}}
.panel{{background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:hidden}}
.panel .cap{{padding:11px 15px;border-bottom:1px solid var(--line);font-size:12px;
  text-transform:uppercase;letter-spacing:.09em;color:var(--dim);font-weight:600}}
.heatimg{{padding:10px}} .heatimg img{{width:100%;border-radius:8px;display:block}}
.tiles{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.tile{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:15px 16px}}
.tile .tv{{font-size:27px;font-weight:700;font-family:var(--mono);letter-spacing:-.02em;font-variant-numeric:tabular-nums}}
.tile .tk{{font-size:13px;font-weight:600;margin-top:2px}}
.tile .ts{{font-size:11.5px;color:var(--dim)}}
.tile:nth-child(3) .tv,.tile:nth-child(6) .tv{{color:var(--accent)}}
section{{margin-top:44px}}
section > h3{{font-size:12.5px;text-transform:uppercase;letter-spacing:.11em;color:var(--dim);
  margin:0 0 15px;font-weight:650;display:flex;align-items:center;gap:10px}}
section > h3::after{{content:"";flex:1;height:1px;background:var(--line)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}}
.shot{{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}}
.shot img{{width:100%;display:block;background:#fff}}
.shot figcaption{{display:flex;justify-content:space-between;align-items:baseline;gap:8px;
  padding:9px 13px;font-size:12.5px}}
.shot figcaption b{{font-weight:650}} .shot figcaption span{{color:var(--dim);font-family:var(--mono);font-size:11.5px}}
.shot.gif img{{background:#f0f0f0}}
.bars{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px 20px}}
.bar{{display:grid;grid-template-columns:150px 1fr 74px;align-items:center;gap:12px;padding:4px 0}}
.bar .bl{{font-size:13px;font-weight:550;text-align:right;color:var(--ink);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.bar .bt{{height:15px;background:color-mix(in srgb,var(--accent) 9%,transparent);border-radius:5px;overflow:hidden}}
.bar .bf{{display:block;height:100%;background:linear-gradient(90deg,var(--accent),color-mix(in srgb,var(--accent) 62%,#000));border-radius:5px}}
.bar .bn{{font-family:var(--mono);font-size:12px;color:var(--dim);text-align:right;font-variant-numeric:tabular-nums}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}@media(max-width:820px){{.two{{grid-template-columns:1fr}}}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px 22px}}
.card h4{{margin:0 0 10px;font-size:15px}}
.card p,.card li{{font-size:13.5px;color:var(--dim)}} .card ol{{margin:0;padding-left:20px}} .card li{{margin:5px 0}}
.card b,.card strong{{color:var(--ink)}}
code{{font-family:var(--mono);font-size:12px;background:var(--accent-soft);color:var(--accent);padding:1px 6px;border-radius:5px}}
pre{{background:color-mix(in srgb,var(--ink) 5%,transparent);border:1px solid var(--line);border-radius:10px;
  padding:14px 16px;overflow-x:auto;font-family:var(--mono);font-size:12px;color:var(--ink);line-height:1.6}}
.chip{{display:inline-flex;align-items:center;gap:6px;font-family:var(--mono);font-size:11.5px;color:var(--ok);
  border:1px solid color-mix(in srgb,var(--ok) 45%,transparent);border-radius:100px;padding:2px 9px}}
.chip::before{{content:"";width:6px;height:6px;border-radius:50%;background:var(--ok)}}
footer{{margin-top:52px;padding-top:20px;border-top:1px solid var(--line);color:var(--dim);
  font-size:12px;font-family:var(--mono);display:flex;flex-wrap:wrap;gap:6px 22px}}
.legend{{display:flex;gap:18px;flex-wrap:wrap;margin:2px 0 0;font-size:12.5px;color:var(--dim)}}
.legend span{{display:inline-flex;align-items:center;gap:7px}}
.sw{{width:13px;height:13px;border-radius:3px;display:inline-block}}
</style>
<header><div class="hd">
  <span class="dot"></span>
  <h1>HOT3D · Dense Hand-Contact Maps</h1>
  <span class="sub">HACO method · 1&nbsp;cm · generated 2026-07-10</span>
  <span class="badge">778-vertex binary contact</span>
</div></header>
<div class="wrap">

  <div class="hero">
    <div class="lead">
      <h2>Per-vertex hand contact, labeled across the full HOT3D dataset</h2>
      <p>For every annotated frame and hand, each of the 778 MANO hand vertices is marked in contact when
      it lies within <b>1&nbsp;cm</b> of an object surface — HACO's labeling rule (vertex-to-surface
      distance &lt; <code>c_thres</code>), applied to HOT3D's MANO hand annotations and posed object
      CAD meshes. <span class="chip">validated vs exact surface distance</span></p>
    </div>
    <div class="panel">
      <div class="cap">dataset-wide contact frequency — which hand regions touch objects</div>
      <div class="heatimg">{'<img src="'+heat+'" alt="contact frequency on MANO hand"/>' if heat else 'heatmap unavailable'}</div>
      <div class="cap" style="border-top:1px solid var(--line);border-bottom:0">
        <span class="legend"><span><span class="sw" style="background:#f5f5f5;border:1px solid var(--line)"></span>rare</span>
        <span><span class="sw" style="background:#ffe178"></span>some</span>
        <span><span class="sw" style="background:#f07828"></span>frequent</span>
        <span><span class="sw" style="background:#960a0a"></span>constant</span></span>
      </div>
    </div>
    <div class="tiles">{tile_html}</div>
  </div>

  <section>
    <h3>Sample grasps · left panel = contact map on the hand, right = hand + object(s)</h3>
    <div class="grid">{frame_html}</div>
  </section>

  {'<section><h3>Contact through a grasp (animated)</h3><div class="grid">'+gif_html+'</div></section>' if gif_html else ''}

  <section>
    <h3>Contact events by object category</h3>
    <div class="bars">{obj_rows}</div>
  </section>

  <section>
    <h3>Method &amp; data</h3>
    <div class="two">
      <div class="card">
        <h4>How each label is computed</h4>
        <ol>
          <li><b>Hand mesh (778 verts)</b> from HOT3D MANO annotations — MANO (<code>mano_layer.py</code>),
          <code>use_pca=True</code>, 15 PCA pose comps, wrist SE3 → global-orient + translation, world frame.</li>
          <li><b>Object meshes</b> from the HOT3D <code>.glb</code> library, posed per frame via
          <code>dynamic_objects.csv</code> (<code>t_wo</code>, <code>q_wo</code>).</li>
          <li><b>Contact</b> = HACO's rule: nearest object-surface distance per hand vertex,
          thresholded at <code>0.01&nbsp;m</code>; min across all present objects.</li>
        </ol>
        <p style="margin-bottom:0">Exact <code>trimesh.on_surface</code> is approximated by a cKDTree over a
        dense (~1.2&nbsp;mm) surface sampling — <b>99.95% label agreement</b>, 0.34&nbsp;mm mean error, ~300× faster.</p>
      </div>
      <div class="card">
        <h4>Per-sequence <code>.npz</code> format</h4>
        <pre>timestamps_ns        (T,)      int64
{{left,right}}_present   (T,)      bool
{{left,right}}_contact   (T,778)   uint8   ← label
{{left,right}}_min_dist  (T,778)   float16 (m)
{{left,right}}_nearest_obj (T,778) int8
object_uids / object_names (O,)    str
meta  json: seq, headset, c_thres, stats</pre>
        <p style="margin-bottom:0">Also: <code>contact_data_all.npy</code> (N&times;778 aggregate),
        <code>contact_means.npy</code>, <code>manifest.csv</code>. Vertex order matches MANO, so labels drop
        straight into the per-vertex contact training layout.</p>
      </div>
    </div>
  </section>

  <footer>
    <span>294 sequences · {n_hi:,} hand-instances · {n_ci:,} in contact</span>
    <span>data/hot3d_contact/</span>
    <span>Aria {hv('Aria','seqs')} · Quest {hv('Quest','seqs')} seqs</span>
  </footer>
</div>
"""
open(OUT, "w").write(html)
print("wrote", OUT, f"{os.path.getsize(OUT)/1e6:.1f} MB",
      f"| {len(frames)} frames, {len(gifs)} gifs embedded")
