# Training

The code that trains the two released contact models on 8-frame clips. Both models share one trainer,
one objective and one data pipeline ([`../data_processing`](../data_processing)). Each `--model` selects a
recipe in [`configs.py`](configs.py). Its values were read back from the `args`, optimizer and scheduler
state that the original runs saved in their own epoch checkpoints.

| | `--model vith` | `--model vitl` |
|---|---|---|
| backbone | V-JEPA 2 ViT-H/16, 256 px (`facebook/vjepa2-vith-fpc64-256`) | V-JEPA 2.1 ViT-L/16, 384 px (`vjepa2_1_vit_large_384`) |
| head | DPT decoder → 778-vertex contact map + 2-bit hand contact | DPT decoder (wider, 3 RCUs per fusion block) → contact map + 2-bit hand contact + dense seg |
| clips | 8 frames, stride 2, ending at the labelled frame | same |
| datasets | 9: HOI4D, ARCTIC, HIC, MOW, H2O, PROX, InterHand2.6M, DECAF, Hi4D (447,471 train / 49,662 val) | all 12 (1,553,353 / 172,877) |
| clips per step | 8 per GPU × 8 GPUs = 64 | 12 per GPU × 32 GPUs (4 nodes) = 384 |
| optimizer | AdamW, β (0.9, 0.95), wd 1e-4; decoder + heads 1e-4, backbone 1e-5 | AdamW, same; decoder + heads 1e-4, backbone 5e-6 |
| schedule | 500-step linear warmup, then cosine to 0 over 10 epochs (6,991 steps/epoch) | 500-step linear warmup, then constant (`warmconst`); 10 epochs (4,045 steps/epoch) |
| precision | bf16 autocast, fp32 loss; backbone gradient-checkpointed; backbone trained from step 0 | same |
| released | `vith.pt` = epoch 1 (val contact-map F1 0.637, binary F1 0.987) | `vitl.pt` = epoch 6 (0.791 / 0.985); epoch 9 reached 0.805 |

The loss is the same for both models (see [`losses.py`](losses.py)). The contact-map term is HACO's objective
restricted to the 778-vertex map: vertex-level class-balanced BCE, plus 0.1 × an L1 term toward the
per-vertex contact frequency, plus a mesh-smoothness term. The 2-bit hand-contact head adds a BCE term. The
ViT-L seg head also gets a BCE + Dice term, but only on samples that come with a mask, and no loader supplies
one. So for the released runs that term is a zero that stays in the graph, which keeps DDP's static graph
valid.

## Files

| | |
|---|---|
| [`train.py`](train.py) | DDP trainer: train, per-epoch validation (overall + per dataset), checkpoints, resume, `--eval_only`, `--smoke` |
| [`configs.py`](configs.py) | the two recipes, with where each number comes from |
| [`losses.py`](losses.py) | the training objective |
| [`models.py`](models.py) | builds the models from `inference/lib/models/`, the same files that load `vith.pt` / `vitl.pt`, so checkpoints work in both directions |
| [`export_checkpoint.py`](export_checkpoint.py) | slims a training checkpoint to the `state_dict` + `epoch` + `val_metrics` file that `inference/run.py --ckpt` takes |
| [`scripts/train_vith.sh`](scripts/train_vith.sh), [`scripts/train_vitl.sh`](scripts/train_vitl.sh) | `torchrun` launchers (single node / multi-node) |

## Setup

**Environment.** Use the `haco_render` environment from [`../inference/envs`](../inference/envs) (Python 3.10,
torch 2.4.1+cu121, transformers 4.53.0, numpy 1.26). The trainer needs no other packages; `tensorboard` is
not used.

**Data.** Run from a directory that contains `data/`, laid out as described in
[`../data_processing/README.md`](../data_processing/README.md). To train on the paper's splits, copy them
into place:

```bash
mkdir -p data/splits_multidata_vjepa && cp <repo>/data_processing/splits/*.json data/splits_multidata_vjepa/
```

The loss also reads two files from HACO's base data release, plus the MANO faces:

```
data/base_data/contact_data/all/contact_data_all.npy     # training-set contact labels (class counts)
data/base_data/contact_data/all/contact_means_all.npy    # per-vertex contact frequency
<repo>/inference/data/base_data/mano_assets.npz          # python inference/tools/make_mano_assets.py
```

**Backbone weights.**

- **vith:** transformers downloads `facebook/vjepa2-vith-fpc64-256` (2.5 GB) on first use. To run offline,
  set `HF_HUB_CACHE` to a cache that already holds it.
- **vitl:** needs the V-JEPA 2.1 ViT-L/16-384 checkpoint,
  `https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt` (5.2 GB). This is the file the
  released run started from. Save it as `$TORCH_HOME/hub/checkpoints/vjepa2_1_vitl_384.pt`, or pass
  `--vjepa2_ckpt`. Its EMA encoder initialises the backbone. The model is built with the vjepa2 repo's
  code, from `$VJEPA2_REPO`; set that up as in
  [`../inference/third_party/README.md`](../inference/third_party/README.md).

## Run

```bash
cd /path/with/data
export PYTHONPATH=<repo>

# check the setup end to end on one GPU: 4 steps, then a tiny validation pass
python <repo>/training/train.py --model vitl --smoke 4 --batch 2 --workers 2

# ViT-H: 8 GPUs, one node
bash <repo>/training/scripts/train_vith.sh

# ViT-L: 4 nodes x 8 GPUs. Run on each node; EXP_DIR on shared storage
NNODES=4 NODE_RANK=<0..3> MASTER_ADDR=<node 0> EXP_DIR=/shared/experiments/vitl \
  bash <repo>/training/scripts/train_vitl.sh
# ...or on one node of 8 GPUs, with the same 384-clip step
bash <repo>/training/scripts/train_vitl.sh --grad_accum 4

# validate a checkpoint (prints per-dataset metrics next to the ones stored in the checkpoint)
python <repo>/training/train.py --model vitl --eval_only --ckpt <repo>/inference/checkpoints/vitl.pt

# use a trained epoch for inference
python <repo>/training/export_checkpoint.py experiments/vitl/checkpoints/haco_vjepa_alldata_epoch6.ckpt vitl_e6.pt
```

Any recipe value can be overridden on the command line: `--datasets`, `--batch`, `--epochs`, `--lr`,
`--backbone_lr`, `--lr_schedule`, `--frames`, `--stride`, `--input_size`. Other flags are `--freeze_backbone 1`
(train the decoder and heads only), `--init_ckpt` (warm start from weights, non-strict) and `--val_max`.
Changing `--epochs` also changes the class-balancing ramp, which runs over the planned number of epochs.

Output goes to `--exp_dir` (default `experiments/<model>`):

- `checkpoints/haco_vjepa_alldata_epoch{E}.ckpt`: weights, optimizer, scheduler, val metrics and args.
  Re-running resumes after the last one.
- `checkpoints/haco_vjepa_alldata_best.ckpt`: best epoch by overall val contact-map F1.
- `results.json` and `train.log`.

## How it relates to the original runs

What was checked against the original runs' checkpoints:

- **Recipe.** Every value in `configs.py` matches the `args` stored in the checkpoint.
- **Optimizer.** `train.py` builds the same AdamW groups (vith 516 backbone / 91 head tensors, vitl 302 / 128).
- **Schedule.** The cosine schedule reproduces the learning rate saved at vith's step 13,982 to the last digit.
- **Weights.** Both original checkpoints load strictly into the models built here.
- **Loss.** It matches HACO's reference implementation on every term to < 1e-6 relative, gradients included.
- **Validation.** `--eval_only` reproduces the per-dataset val F1 stored in `vitl.pt` exactly on EPIC-Contact
  (0.7120), and on PROX to within 0.003. The PROX gap comes from sub-pixel rounding in the crop affine.

Differences from the original runs:

- **`warmconst` is reconstructed.** The original ViT-L trainer is not preserved. Its checkpoint records
  `lr_schedule='warmconst'` and a learning rate still at its peak at step 28,315. Here `warmconst` uses the
  same 500-step warmup as the cosine schedule its trainer descended from, then holds the rate constant.
- **Unused options dropped.** The ViT-L trainer had a few options that the released run left at their
  off settings: `freeze_epochs=0`, `unfreeze_top_blocks=0`, `contact_only=False`. They are not carried over.
- **`best.ckpt` fixed.** In the original 12-dataset runs, `best.ckpt` stopped updating after epoch 2. Here the
  best epoch is stored in every epoch checkpoint and survives a resume.
- **Validation padding kept.** As before, `DistributedSampler` pads the val set to a multiple of the world size.
  The few repeated clips come from the first dataset in the roster, HOI4D, so its val count reads slightly
  above the split size (vitl.pt: 12,014 vs 11,995).
- **MOW and HIC val numbers in the checkpoints are not reproducible.** The original loader looked for a
  `val.json` split file in these two datasets, found none, and silently fell back to its train-mode reader.
  So the original runs scored MOW and HIC val crops with random training augmentation, including the noise
  step that turns about 30% of crops near-black, and redrew it every epoch. `data_processing` scores them
  on deterministic crops, as it does every other dataset. When the original reader is forced into its
  deterministic mode, its crops equal ours exactly. So the MOW and HIC val F1 from `--eval_only` will differ
  from the values stored in the checkpoints. For `vitl.pt`, HIC scores 0.33 here vs 0.30 stored; for
  `vith.pt`, MOW scores 0.53 vs 0.41. These are 59 of 172,877 val clips, and training is unaffected: the
  splits, labels, sample order and augmentation distribution are the same.
- **Training is not bit-reproducible.** Augmentation draws come from a different random stream than in
  the original loaders. The distribution is the same, but the individual draws are not.
