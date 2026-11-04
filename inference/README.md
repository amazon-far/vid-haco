# Vid-HACO inference

Video in, hand-contact video out. Give it any video with a hand in it: it tracks the hand in 3D, predicts
which of the 778 MANO mesh vertices are touching something, and renders a four-panel visualisation — the
input frame with the hand mesh overlaid, the contact map on a rest-pose hand (dorsal + palmar), and the
same map on the actual grasp pose.

```bash
./run.py --video my_clip.mp4 --out-dir out/my_clip --model vitl
```

## Setup

1. **Environments.** Two conda environments are needed because the tracker pins an old torch:
   `haco_render` (contact model + rendering) and `hawor` (tracker). Exact package lists are in
   [`envs/`](envs) (`*_explicit.txt` for `conda create --file`, `*_pip.txt` for pip).
2. **Third-party code and weights** — V-JEPA 2, HaWoR and the WiLoR hand detector: see
   [`third_party/README.md`](third_party/README.md). You convert the detector to ONNX once, and the pipeline
   runs it with onnxruntime.
3. **MANO.** Register at [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de), download the MANO model and
   put `MANO_RIGHT.pkl` in `data/base_data/human_models/mano/`. Then build the asset file the render
   stage reads (template mesh, faces, joint regressor, skinning weights):
   ```bash
   python tools/make_mano_assets.py --mano data/base_data/human_models/mano/MANO_RIGHT.pkl \
                                    --out data/base_data/mano_assets.npz
   ```
   (HaWoR needs its own copy of MANO in its checkout; see its README.)
4. **Checkpoints:** not yet released. The checkpoints need further legal approval and will be released once approved. See [`checkpoints/`](checkpoints).
5. **Config:** `cp config.env.example config.env` and fill in the paths, then run `./doctor.py`, which
   verifies the tree, both environments, the tracker, the checkpoints, ffmpeg and the GPUs.

## What it produces

```
out/my_clip/
├── my_clip_vitl_meshtop.mp4     the visualisation (1500x1000, source frame rate)
├── my_clip_vitl_combined.npz    every prediction, as arrays — the thing to build on
├── run_summary.json             what ran, with which checkpoint, over which frames
├── samples/                     a few PNG frames
├── logs/                        one log per stage
└── work/                        intermediates (see "Re-running cheaply")
```

Per frame the combined npz carries `contact_prob [N,778]` (per-vertex probability), `hand_contact [N,2]`,
the hand mesh `verts`, wrist translation `cam_t`, `focal`, `bbox_sm`, `found`, plus `frame_idx`, `fps`,
`thr`, `hand_side`, `img_wh`, and the tracker's world/camera transforms and MANO parameters.

> **Reading `hand_contact`:** column 1 is the tracked hand; column 0 carries no information. Use
> `contact_prob` or `hand_contact[:, 1]`.

## Choosing a model

| `--model` | checkpoint | best at | backbone | contact-map F1 | binary-contact F1 |
|---|---|---|---|---|---|
| `vitl` (default) | `vitl.pt` | **contact map estimation** — which vertices touch | V-JEPA 2.1 ViT-L/384 + DPT, 12-dataset run, epoch 6 | **0.791** | 0.985 |
| `vith` | `vith.pt` | **binary contact detection** — is the hand touching | V-JEPA 2 ViT-H/256 + DPT, 8-frame-sequence run, epoch 1 | 0.637 | **0.987** |

Use `vitl` for the dense per-vertex contact map and `vith` for a per-frame contact / no-contact decision.

Both load with `strict=True`, so a mismatched checkpoint fails immediately. They use different Stage-B
runners and input resolutions (384 vs 256), which `run.py` handles. Contact **density** is a property of
the training run: the ViT-H predicts fewer contact vertices per frame than the ViT-L on the same clip; if
you compare them, compare at matched density or tune `--thr`.

## Which hand

The contact model is *right-canonical*: training mirrors every left hand, and the tracker only does right
hands. A left-handed video is therefore mirrored first and the output stays in mirror space (the banner
still names the true side). `--hand auto` (default) decides from the detector's handedness; override with
`--hand left|right`. One hand per run — run twice and combine the npz files for both.

## Common adjustments

```bash
--thr 0.45                          # contact threshold (default 0.52); lower = more contact
--mesh-alpha 0.35                   # overlay opacity (default 0.6)
--start-frame 300 --end-frame 900   # only this range of source frames
--fps 30                            # force an output frame rate (default: the source's)
--filter none                       # skip track smoothing (default: zero-phase 2 Hz Butterworth)
--gpus 0,1,2                        # restrict to these GPUs
--mem-frac 0.9                      # per-process GPU memory cap
--skip-render                       # stop after the npz
--dry-run                           # print the plan and exit
--set KEY=VALUE                     # override any config.env key for one run
```

## Re-running cheaply

Stages are checkpointed; `--resume` skips whatever already finished. Video preparation, tracking and track
filtering do not depend on the contact model, so switching `--model` with `--resume` only re-runs
inference, combine and render.

## How it works

```
   prepare      constant frame rate; mirror if the hand is left; trim to --start/--end
      |
   Stage A      detect the hands (ONNX detector), EMA-smooth the crop box
      |
      +--------------------------------+                 <-- run concurrently
      |                                |
   HaWoR tracking                   Stage B   contact inference (8-frame clips), sharded across GPUs
   (Stage A's detections -> tracks     |
    -> DROID-SLAM -> motion -> infill) |
      |                                |
   Butterworth filter <----------------+
   (zero-phase, 2 Hz)
      |
   combine     contact + filtered track -> one npz over one shared frame interval
      |
   render      four-panel mesh-top video, sharded across GPUs, concatenated
```

* **Frame alignment.** The prepare stage re-stamps timestamps from the frame ordinal so the rate is constant
  without dropping or duplicating frames; trimming happens there too, so the prepared video starts at frame 0.
* **Range reconciliation.** The tracker and Stage A both cover the whole prepared video (a frame where the
  detector misses the hand reuses the last detected box); the track is sliced to Stage A's range before
  combining.
* **Sharded rendering.** The contact-mask smoothing is a causal EMA, so each shard warms up over 48 rows
  before its window; shards are concatenated losslessly.
* **Filtering** touches only the rendered pose; contact predictions are untouched.
