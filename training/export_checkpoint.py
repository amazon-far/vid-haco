#!/usr/bin/env python3
"""Slim a training checkpoint into the file `inference/run.py --ckpt` loads: state_dict + epoch +
val_metrics, no optimizer or scheduler state (the released vith.pt / vitl.pt have this form).

    python training/export_checkpoint.py experiments/vitl/checkpoints/haco_vjepa_alldata_epoch6.ckpt vitl.pt
"""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('out')
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False, mmap=True)
    slim = {k: ck[k] for k in ('state_dict', 'epoch', 'val_metrics') if k in ck}
    torch.save(slim, a.out)
    vm = slim.get('val_metrics') or {}
    print(f"wrote {a.out}: epoch {slim.get('epoch')}, {len(slim['state_dict'])} tensors, "
          f"val C-F1 {vm.get('c_f1', float('nan')):.4f}, bin F1 {vm.get('bin_f1', float('nan')):.4f}")


if __name__ == '__main__':
    main()
