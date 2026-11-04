"""
HACO_VJEPA (CONTACT-ONLY): V-JEPA 2.1 ViT-Gigantic/16-384 (frozen) + the ORIGINAL 46M DPT head
predicting the 778-vertex contact map + 2-bit hand contact. NO segmentation head / output.

This is the exact head architecture that produced experiments_vjepa_vitG_alldata epoch-0
(the 0.629 checkpoint), reconstructed so that checkpoint loads with missing=0 / unexpected=0.
It is kept in a SEPARATE file from lib/models/model_vjepa.py (which now carries the heavier
multi-task DPT + seg head for the parked multi-task experiment) so neither run disturbs the other.

Backbone + token/tap logic are reused verbatim from model_vjepa (frozen encoder, taps [11,23,37,47],
last-tubelet target-frame 24x24 maps). Only the head differs: original DPTDecoder (feats=256,
readout 256/512/1024/1024, 2-RCU RefineNet fusion) -> multi-scale pooled descriptor (1024) ->
contact MLP (->778) + hand MLP (->2).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.model_vjepa import (
    _build_vjepa21_gigantic_encoder, _ResidualConvUnit, VJEPA2_MODEL_ID, VJEPA2_CKPT,
)


# ------------------------------------------------------------------ ORIGINAL DPT blocks
class _FeatureFusionBlock(nn.Module):
    """RefineNet-style fusion (ORIGINAL 2-RCU variant): add a resampled skip, refine, upsample x2."""
    def __init__(self, feats, upsample=True):
        super().__init__()
        self.upsample = upsample
        self.rcu1 = _ResidualConvUnit(feats)
        self.rcu2 = _ResidualConvUnit(feats)
        self.out_conv = nn.Conv2d(feats, feats, 1, 1, 0)

    def forward(self, x, skip=None):
        if skip is not None:
            x = x + self.rcu1(skip)
        x = self.rcu2(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.out_conv(x)
        return x


class DPTDecoder(nn.Module):
    """Original DPT-style decoder over 4 hooked ViT-Gigantic layers -> pyramid [24,48,96,96]."""
    def __init__(self, hidden=1664, feats=256, readout_chans=(256, 512, 1024, 1024), grid=24):
        super().__init__()
        self.feats = feats
        self.grid = grid
        self.proj = nn.ModuleList([nn.Conv2d(hidden, c, 1, 1, 0) for c in readout_chans])
        self.resample = nn.ModuleList([
            nn.ConvTranspose2d(readout_chans[0], readout_chans[0], 4, 4, 0),
            nn.ConvTranspose2d(readout_chans[1], readout_chans[1], 2, 2, 0),
            nn.Conv2d(readout_chans[2], readout_chans[2], 3, 1, 1),
            nn.Conv2d(readout_chans[3], readout_chans[3], 3, 2, 1),
        ])
        self.to_feats = nn.ModuleList([nn.Conv2d(c, feats, 3, 1, 1, bias=False) for c in readout_chans])
        self.fuse3 = _FeatureFusionBlock(feats, upsample=True)
        self.fuse2 = _FeatureFusionBlock(feats, upsample=True)
        self.fuse1 = _FeatureFusionBlock(feats, upsample=True)
        self.fuse0 = _FeatureFusionBlock(feats, upsample=False)
        self.out_conv = nn.Sequential(
            nn.Conv2d(feats, feats, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feats),
            nn.ReLU(inplace=True),
        )

    def forward(self, maps):
        f = [self.to_feats[i](self.resample[i](self.proj[i](m))) for i, m in enumerate(maps)]
        p3 = self.fuse3(f[3])
        p2 = self.fuse2(p3, skip=f[2])
        p1 = self.fuse1(p2, skip=f[1])
        p0 = self.fuse0(p1, skip=f[0])
        dense = self.out_conv(p0)
        return [p3, p2, p1, dense]


# ------------------------------------------------------------------ contact-only model
class HACO_VJEPA(nn.Module):
    def __init__(self, freeze_backbone=True, binary_out=2, hook_layers=None,
                 dpt_feats=256, model_id=VJEPA2_MODEL_ID, torch_dtype=torch.float32,
                 load_backbone_weights=True, ckpt_path=VJEPA2_CKPT, img_size=384):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.freeze_backbone = freeze_backbone
        self.binary_out = binary_out
        self.model_id = model_id
        self.img_size = int(img_size)

        # ----- backbone: V-JEPA 2.1 ViT-Gigantic/16-384 (reused, unchanged) -----
        self.backbone = _build_vjepa21_gigantic_encoder(
            ckpt_path=ckpt_path, load_weights=load_backbone_weights)
        self.hidden = int(self.backbone.embed_dim)
        self.tubelet = int(self.backbone.tubelet_size)
        self.patch = int(self.backbone.patch_size)
        self.grid = self.img_size // self.patch
        self.n_layers = len(self.backbone.blocks)
        if hook_layers is None:
            hook_layers = tuple(self.backbone.hierarchical_layers)
        self.hook_layers = tuple(hook_layers)
        self.backbone.out_layers = list(self.hook_layers)
        self.backbone.attn_out = False

        self._grad_ckpt = False
        self.set_backbone_trainable(not freeze_backbone)

        # ----- ORIGINAL 46M DPT decoder + contact/hand heads (NO seg) -----
        self.decoder = DPTDecoder(hidden=self.hidden, feats=dpt_feats, grid=self.grid)
        desc_dim = dpt_feats * 4
        self.desc_norm = nn.LayerNorm(desc_dim)
        self.contact_head = nn.Sequential(
            nn.Linear(desc_dim, 1024), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(1024, 1024), nn.GELU(),
            nn.Linear(1024, 778),
        )
        self.hand_contact_head = nn.Sequential(
            nn.Linear(desc_dim, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, self.binary_out),
        )
        self._init_heads()

    def _init_heads(self):
        for m in list(self.contact_head) + list(self.hand_contact_head):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    # ---------- freeze control (same semantics as model_vjepa) ----------
    def set_backbone_trainable(self, trainable):
        self.freeze_backbone = not trainable
        self.backbone.requires_grad_(bool(trainable))
        if trainable:
            self.backbone.train()
            self.enable_gradient_checkpointing(True)
        else:
            self.backbone.eval()
            self.enable_gradient_checkpointing(False)
        return self

    def enable_gradient_checkpointing(self, flag=True):
        self._grad_ckpt = bool(flag)
        self.backbone.use_activation_checkpointing = bool(flag)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    # ---------- token -> target-frame spatial map ----------
    def _last_tubelet_map(self, tokens):
        B, N, C = tokens.shape
        s = self.grid * self.grid
        tub = tokens[:, N - s:N, :]
        return tub.transpose(1, 2).reshape(B, C, self.grid, self.grid)

    def _encode(self, clip):
        video = clip.permute(0, 2, 1, 3, 4).contiguous()   # [B,T,3,H,W] -> [B,3,T,H,W]
        if self.freeze_backbone:
            with torch.no_grad():
                outs = self.backbone(video)
            outs = [o.detach() for o in outs]
        else:
            outs = self.backbone(video)
        return [self._last_tubelet_map(o.float()) for o in outs]

    # ---------- forward (contact + binary only) ----------
    def forward(self, inputs, mode="test"):
        clip = inputs["input"]["image"].to(self.device)
        maps = self._encode(clip)
        pyramid = self.decoder(maps)
        desc = torch.cat([p.mean(dim=(2, 3)) for p in pyramid], dim=1)
        desc = self.desc_norm(desc)
        contact_out = self.contact_head(desc)
        hand_contact_out = self.hand_contact_head(desc)
        return dict(contact_out=contact_out, hand_contact_out=hand_contact_out)

    # ---------- introspection ----------
    def trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def frozen_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)
