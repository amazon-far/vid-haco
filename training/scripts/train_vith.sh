#!/usr/bin/env bash
# V-JEPA 2 ViT-H/256 + DPT on 8-frame clips, 9 datasets, 8 GPUs on one node (the released vith run:
# 8 clips/GPU = 64 per step, cosine schedule, 10 planned epochs; vith.pt is epoch 1).
#
#   DATA_ROOT  directory holding data/ (default: current directory)
#   EXP_DIR    output directory (default: experiments/vith); set it here, not as --exp_dir
#   NPROC      GPUs on this node (default 8)
#   extra arguments go to train.py, e.g. --epochs 4 or --batch 4 --grad_accum 2
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${DATA_ROOT:-$PWD}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" TOKENIZERS_PARALLELISM=false
EXP_DIR="${EXP_DIR:-experiments/vith}"
mkdir -p "$EXP_DIR"
for a in "$@"; do [[ "$a" == --exp_dir* ]] && { echo "set EXP_DIR=... instead of passing --exp_dir" >&2; exit 2; }; done
torchrun --standalone --nproc_per_node="${NPROC:-8}" "$REPO/training/train.py" \
  --model vith --exp_dir "$EXP_DIR" "$@" 2>&1 | tee -a "$EXP_DIR/stdout.log"
