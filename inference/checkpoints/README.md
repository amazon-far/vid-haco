# checkpoints

The two contact models that `run.py --model` selects.

> **Note:** The checkpoints need further legal approval and will be released once approved.

| file | `--model` | best at | what it is | contact-map F1 | binary-contact F1 |
|---|---|---|---|---|---|
| `vitl.pt` | `vitl` | **contact map estimation** (which vertices touch) | V-JEPA 2.1 ViT-L/384 + DPT, 12-dataset run, epoch 6. 1.65 GB. | **0.791** | 0.985 |
| `vith.pt` | `vith` | **binary contact detection** (is the hand touching?) | V-JEPA 2 ViT-H/256 + DPT, 8-frame-sequence run, epoch 1. 2.59 GB. | 0.637 | **0.987** |

Each file holds `state_dict` + `epoch` + `val_metrics` (no optimizer state).

Both load with `strict=True` in Stage B, which prints the epoch and validation metric it found — check
that line in `out/<clip>/logs/stageB_0.log` if you ever doubt which weights ran.

`config.env` (`CKPT_VITL`, `CKPT_VITH`) points at these files; `--ckpt` overrides it for one run.
