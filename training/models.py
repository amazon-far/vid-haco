"""
Builds the two contact models for training from the SAME module files the inference pipeline loads the
released checkpoints with (inference/lib/models/), so a checkpoint written by train.py loads strict in
Stage B and vice versa.

  vith  inference/lib/models/model_vjepa_seq.py       transformers VJEPA2Model (facebook/vjepa2-vith-fpc64-256)
                                                      + DPT decoder + contact / hand-contact heads
  vitl  inference/lib/models/model_vjepa_vitl_seq.py  V-JEPA 2.1 ViT-L/16-384 (vjepa2 repo builder)
                                                      + DPT decoder + seg / contact / hand-contact heads

Backbone weights for training from scratch:
  vith  downloaded by transformers on first use (or HF_HUB_CACHE / HF_HUB_OFFLINE=1 with a local cache)
  vitl  the V-JEPA 2.1 checkpoint, $TORCH_HOME/hub/checkpoints/vjepa2_1_vitl_384.pt (its EMA
        `target_encoder` is used) or the slim vjepa2_1_vitl_384_encoder.pt; override with --vjepa2_ckpt.
        The vjepa2 repo itself is imported from $VJEPA2_REPO (default inference/third_party/vjepa2_repo).
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_INFERENCE = os.path.join(REPO, 'inference')
if _INFERENCE not in sys.path:
    sys.path.insert(0, _INFERENCE)


def build_model(name, model_id, freeze_backbone=False, input_size=None, vjepa2_ckpt=None,
                load_backbone_weights=True, log=print):
    """-> nn.Module. `load_backbone_weights=False` skips reading pretrained ViT-L weights (use when a
    contact checkpoint is loaded right after, e.g. --eval_only)."""
    if name == 'vith':
        from lib.models.model_vjepa_seq import HACO_VJEPA
        model = HACO_VJEPA(freeze_backbone=freeze_backbone, binary_out=2, model_id=model_id)
        if not freeze_backbone:
            # transformers' own per-layer checkpointing; non-reentrant so DDP(static_graph) is happy
            model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            log("[model] vith: gradient checkpointing on the V-JEPA 2 encoder (use_reentrant=False)")
    elif name == 'vitl':
        from lib.models import model_vjepa_vitl_seq as m
        kw = dict(freeze_backbone=freeze_backbone, binary_out=2, model_id=model_id,
                  img_size=input_size or 384, n_force=0, load_backbone_weights=load_backbone_weights)
        if load_backbone_weights:
            kw['ckpt_path'] = vjepa2_ckpt or m.VJEPA2_CKPT
        # n_force=0: the released 12-dataset run had no force head. set_backbone_trainable() inside the
        # constructor turns on the encoder's per-block activation checkpointing when unfrozen.
        model = m.HACO_VJEPA(**kw)
    else:
        raise ValueError(f"unknown model {name!r} (expected 'vith' or 'vitl')")
    return model


def split_params(model):
    """(backbone params, head params) among the trainable ones — the two AdamW groups."""
    bb, head = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (bb if n.startswith('backbone.') else head).append(p)
    return bb, head
