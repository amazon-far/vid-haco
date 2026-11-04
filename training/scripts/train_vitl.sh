#!/usr/bin/env bash
# V-JEPA 2.1 ViT-L/384 + DPT on 8-frame clips, 12 datasets (the released vitl run: 12 clips/GPU on
# 32 GPUs = 384 per step, warmup-then-constant schedule, 10 epochs; vitl.pt is epoch 6).
#
# Multi-node: run this on every node with the same MASTER_ADDR and NNODES, and NODE_RANK=0..NNODES-1.
#   NNODES=4 NODE_RANK=<i> MASTER_ADDR=<node-0 ip> bash training/scripts/train_vitl.sh
# One node: NNODES=1 (default). Pass --grad_accum 4 to keep the 384-clip step on 8 GPUs.
#
#   DATA_ROOT    directory holding data/ (default: current directory)
#   EXP_DIR      output directory (default: experiments/vitl) -- shared storage when multi-node;
#                set it here, not as --exp_dir
#   TORCH_HOME   holds hub/checkpoints/vjepa2_1_vitl_384.pt (the V-JEPA 2.1 weights), or pass --vjepa2_ckpt
#   VJEPA2_REPO  vjepa2 checkout (default inference/third_party/vjepa2_repo)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${DATA_ROOT:-$PWD}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
EXP_DIR="${EXP_DIR:-experiments/vitl}"
NNODES="${NNODES:-1}"
mkdir -p "$EXP_DIR"
for a in "$@"; do [[ "$a" == --exp_dir* ]] && { echo "set EXP_DIR=... instead of passing --exp_dir" >&2; exit 2; }; done
if [ "$NNODES" -gt 1 ]; then
  LAUNCH=(--nnodes="$NNODES" --node_rank="${NODE_RANK:?set NODE_RANK}" --master_addr="${MASTER_ADDR:?set MASTER_ADDR}"
          --master_port="${MASTER_PORT:-29500}")
else
  LAUNCH=(--standalone)
fi
torchrun "${LAUNCH[@]}" --nproc_per_node="${NPROC:-8}" "$REPO/training/train.py" \
  --model vitl --exp_dir "$EXP_DIR" "$@" 2>&1 | tee -a "$EXP_DIR/stdout.node${NODE_RANK:-0}.log"
