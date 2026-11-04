"""
HACO_VJEPA: V-JEPA 2.1 ViT-L/16 (384) video backbone + DPT decoder + dual contact head.

Backbone  (V-JEPA 2.1 ViT-L/16-384, built with `vjepa2_1_vit_large_384` from the vjepa2 repo in
  third_party/vjepa2_repo)
  embed_dim=1024, patch=16, tubelet=2, img=384x384, RoPE (interpolate_rope). For inference every backbone
  weight comes from the contact checkpoint; for training from scratch the encoder is initialised from
  the V-JEPA 2.1 checkpoint's `target_encoder` (the EMA teacher). Runs on torch SDPA
  (F.scaled_dot_product_attention). FROZEN by default; the freeze schedule in the trainer flips it
  trainable from epoch >= freeze_epochs. (This file started as a copy of the ViT-g variant,
  model_vjepa_vitg_seq.py; only the backbone builder differs.)

Input / tokens
  A clip of T RGB frames at 384x384: the trainer/loader emit [B, T, 3, 384, 384]; we permute to the
  encoder's [B, 3, T, 384, 384]. Conv3d patch-embed (tubelet=2) yields T'=T/2 temporal x 24x24 spatial
  tokens, ordered TEMPORAL-MAJOR (flatten(2).transpose -> token = t'*576 + h*24 + w). RoPE handles
  variable T (interpolate_rope=True), so no learned temporal pos-emb interpolation is needed.

Target-frame feature
  We predict contact for the LAST (target) frame of the clip -> take the LAST temporal tubelet's
  tokens (indices [(T'-1)*576 : T'*576]) and reshape to a 24x24 grid -> [B, hidden, 24, 24].

DPT decoder (TRAINABLE)
  We tap the encoder's 4 native hierarchical layers (`backbone.hierarchical_layers`, its distillation
  taps, evenly spanning shallow->deep) via encoder.out_layers -> a list of 4 LayerNorm'd token maps
  [B, N, 1024]. For each we extract the target-frame 24x24x1024 map, DPT-"reassemble" to 4 pyramid
  resolutions (96, 48, 24, 12) with increasing readout width (384, 768, 1536, 1536), unify to 384
  channels, then fuse top-down with RefineNet-style feature-fusion blocks (3 residual conv units each)
  into a multi-scale feature pyramid. A multi-scale context descriptor (global-avg-pool of every
  pyramid stage, concatenated) feeds the heads.

Heads (match the existing HACO output dict so losses / eval are reusable)
  * contact-map head: descriptor -> LayerNorm -> MLP -> 778 logits -> contact_out       [B, 778]
  * binary head:      descriptor -> MLP -> `binary_out` logits    -> hand_contact_out    [B, 2]
  * force head:       descriptor -> MLP -> n_force values       -> force_out           [B, 6]
                      (thumb,index,middle,ring,pinky,palm in LOG1P newtons; FEEL-only supervision;
                      absent from the released checkpoint, so load_state_dict drops it)

forward(inputs, mode) -> dict(contact_out=[B,778], hand_contact_out=[B,2]).
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----- V-JEPA 2.1 hub repo + checkpoint (overridable via env) -----
VJEPA2_REPO = os.environ.get("VJEPA2_REPO", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "third_party", "vjepa2_repo"))
_CKPT_DIR = os.path.join(os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch")), "hub", "checkpoints")
# Prefer the slim encoder-only checkpoint (7.4 GB) if present -- much lighter for 8-rank DDP than the
# full 30 GB training checkpoint (which also carries predictor + optimizer + EMA). Falls back to the
# full checkpoint (we extract `target_encoder` from it) if the slim file is absent.
_SLIM = os.path.join(_CKPT_DIR, "vjepa2_1_vitl_384_encoder.pt")
_FULL = os.path.join(_CKPT_DIR, "vjepa2_1_vitl_384.pt")
VJEPA2_CKPT = os.environ.get("VJEPA2_CKPT", _SLIM if os.path.isfile(_SLIM) else _FULL)
# Kept for backward-compat with the trainer's --model_id flag; a plain identifier now.
VJEPA2_MODEL_ID = "vjepa2.1-vitl-384"


def _build_vjepa21_giant_encoder(ckpt_path=VJEPA2_CKPT, load_weights=True, verbose=True):
    """Build the V-JEPA 2.1 ViT-Large/16-384 encoder (embed_dim 1024).

    NOTE: kept the giant-era function name so the rest of this file (cloned from the ViT-g variant)
    is byte-for-byte identical. The ONLY difference vs the giant is the backbone builder called here
    (vjepa2_1_vit_large_384). Everything downstream reads self.backbone.embed_dim, so it adapts to
    the 1024-dim ViT-L automatically. For contact INFERENCE we always load a trained contact checkpoint
    (load_backbone_weights=False), so the encoder .pt below is never read — no ViT-L pretrained
    encoder download is needed; the contact checkpoint supplies all backbone weights.
    """
    if VJEPA2_REPO not in sys.path:
        sys.path.insert(0, VJEPA2_REPO)
    from src.hub.backbones import vjepa2_1_vit_large_384, _clean_backbone_key

    encoder, predictor = vjepa2_1_vit_large_384(pretrained=False)
    del predictor  # V-JEPA 2.1 predictor is unused for the contact task

    if load_weights:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"V-JEPA 2.1 ViT-L encoder checkpoint not found at {ckpt_path}. "
                f"(Only needed for from-scratch training; contact inference loads a contact checkpoint "
                f"and passes load_backbone_weights=False.)"
            )
        sd = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
        if isinstance(sd, dict) and any(k in sd for k in ("target_encoder", "ema_encoder", "encoder")):
            key = "target_encoder" if "target_encoder" in sd else ("ema_encoder" if "ema_encoder" in sd else "encoder")
            enc_sd = _clean_backbone_key({k: v for k, v in sd[key].items()})
            src = f"full ckpt['{key}']"
        else:  # slim: already a clean encoder state_dict
            enc_sd = _clean_backbone_key({k: v for k, v in sd.items()})
            src = "slim encoder ckpt"
        missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
        if verbose:
            print(f"[vjepa21] loaded encoder weights from {src} ({os.path.basename(ckpt_path)}): "
                  f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    return encoder


# ------------------------------------------------------------------ DPT building blocks
class _ResidualConvUnit(nn.Module):
    """Pre-activation residual conv unit (RefineNet/DPT)."""
    def __init__(self, feats):
        super().__init__()
        self.conv1 = nn.Conv2d(feats, feats, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(feats)
        self.conv2 = nn.Conv2d(feats, feats, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(feats)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(self.conv1(self.act(x)))
        out = self.bn2(self.conv2(self.act(out)))
        return out + x


class _FeatureFusionBlock(nn.Module):
    """RefineNet-style fusion: refine a resampled skip (1 RCU), add it, then refine with (n_rcu-1)
    RCUs, optionally upsample x2. `n_rcu` controls fusion depth (heavier decoder -> larger n_rcu)."""
    def __init__(self, feats, upsample=True, n_rcu=2):
        super().__init__()
        self.upsample = upsample
        self.skip_rcu = _ResidualConvUnit(feats)
        self.rcus = nn.ModuleList([_ResidualConvUnit(feats) for _ in range(max(1, n_rcu - 1))])
        self.out_conv = nn.Conv2d(feats, feats, 1, 1, 0)

    def forward(self, x, skip=None):
        if skip is not None:
            x = x + self.skip_rcu(skip)
        for rcu in self.rcus:
            x = rcu(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.out_conv(x)
        return x


class DPTDecoder(nn.Module):
    """Powerful DPT-style decoder over 4 hooked ViT-giant layers.

    Inputs : list of 4 spatial maps [B, hidden, grid, grid] (shallow -> deep), grid=24 at 384-res.
    Output : a multi-scale feature pyramid (list of [B, feats, R, R] maps at R in {96,48,24,24}).
    """
    def __init__(self, hidden=1408, feats=256, readout_chans=(256, 512, 1024, 1024), grid=24, n_rcu=2):
        super().__init__()
        self.feats = feats
        self.grid = grid
        c0, c1, c2, c3 = readout_chans
        # 1x1 projections from ViT hidden dim -> per-stage readout channels (increasing with depth)
        self.proj = nn.ModuleList([nn.Conv2d(hidden, c, 1, 1, 0) for c in readout_chans])
        # DPT reassemble: resample each stage to a distinct pyramid resolution from the grid base
        #   stage0 (shallow, L11): grid -> grid*4 (x4, ConvTranspose k4s4)   [96]
        #   stage1 (       , L23): grid -> grid*2 (x2, ConvTranspose k2s2)   [48]
        #   stage2 (       , L37): grid -> grid   (identity 3x3 conv)        [24]
        #   stage3 (deep,   L47): grid -> grid/2 (stride-2 conv)             [12]
        self.resample = nn.ModuleList([
            nn.ConvTranspose2d(c0, c0, 4, 4, 0),
            nn.ConvTranspose2d(c1, c1, 2, 2, 0),
            nn.Conv2d(c2, c2, 3, 1, 1),
            nn.Conv2d(c3, c3, 3, 2, 1),
        ])
        # unify every stage to `feats` channels
        self.to_feats = nn.ModuleList([nn.Conv2d(c, feats, 3, 1, 1, bias=False) for c in readout_chans])
        # top-down RefineNet fusion (deep -> shallow), n_rcu RCUs per block
        self.fuse3 = _FeatureFusionBlock(feats, upsample=True, n_rcu=n_rcu)   # 12 -> 24
        self.fuse2 = _FeatureFusionBlock(feats, upsample=True, n_rcu=n_rcu)   # 24 -> 48 (skip stage2@24)
        self.fuse1 = _FeatureFusionBlock(feats, upsample=True, n_rcu=n_rcu)   # 48 -> 96 (skip stage1@48)
        self.fuse0 = _FeatureFusionBlock(feats, upsample=False, n_rcu=n_rcu)  # 96 -> 96 (skip stage0@96)
        self.out_conv = nn.Sequential(
            nn.Conv2d(feats, feats, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feats),
            nn.ReLU(inplace=True),
        )

    def forward(self, maps):
        # maps[i]: [B, hidden, grid, grid]
        f = [self.to_feats[i](self.resample[i](self.proj[i](m))) for i, m in enumerate(maps)]
        # f[0]:96  f[1]:48  f[2]:24  f[3]:12
        p3 = self.fuse3(f[3])              # 12 -> 24
        p2 = self.fuse2(p3, skip=f[2])     # 24(+skip@24) -> 48
        p1 = self.fuse1(p2, skip=f[1])     # 48(+skip@48) -> 96
        p0 = self.fuse0(p1, skip=f[0])     # 96(+skip@96) -> 96
        dense = self.out_conv(p0)          # [B, feats, 96, 96]
        return [p3, p2, p1, dense]         # multi-scale pyramid (24, 48, 96, 96)


# ------------------------------------------------------------------ full model
class HACO_VJEPA(nn.Module):
    def __init__(self, freeze_backbone=True, binary_out=2, hook_layers=None,
                 dpt_feats=384, dpt_readout=(384, 768, 1536, 1536),
                 model_id=VJEPA2_MODEL_ID, torch_dtype=torch.float32,
                 load_backbone_weights=True, ckpt_path=VJEPA2_CKPT, img_size=384, n_force=6, force_hurdle=True):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.freeze_backbone = freeze_backbone
        self.binary_out = binary_out
        self.model_id = model_id
        self.img_size = int(img_size)

        # ----- backbone: V-JEPA 2.1 ViT-giant/16-384 -----
        self.backbone = _build_vjepa21_giant_encoder(
            ckpt_path=ckpt_path, load_weights=load_backbone_weights)
        self.hidden = int(self.backbone.embed_dim)                 # 1408
        self.tubelet = int(self.backbone.tubelet_size)             # 2
        self.patch = int(self.backbone.patch_size)                 # 16
        self.grid = self.img_size // self.patch                    # 24 tokens per side
        self.n_layers = len(self.backbone.blocks)                  # 48
        # 4 DPT taps -> use the encoder's native hierarchical (distillation) layers if not given
        if hook_layers is None:
            hook_layers = tuple(self.backbone.hierarchical_layers)  # [9, 19, 29, 39]
        self.hook_layers = tuple(hook_layers)
        # drive the encoder to return the 4 tapped, LayerNorm'd token maps as a list
        self.backbone.out_layers = list(self.hook_layers)
        self.backbone.attn_out = False

        self._grad_ckpt = False
        self._surgical = False          # surgical (top-K) fine-tune mode
        self._surgical_split = 0        # index of the first TRAINABLE block when surgical
        self.set_backbone_trainable(not freeze_backbone)

        # ----- HEAVY DPT decoder (trainable) -- the backbone is frozen so we spend the budget here.
        #   feats=384 (up from 256), richer readout widths (384/768/1536/1536), 3 RCUs per fusion.
        self.decoder = DPTDecoder(hidden=self.hidden, feats=dpt_feats, grid=self.grid,
                                  readout_chans=dpt_readout, n_rcu=3)
        # multi-scale descriptor = concat of global-avg-pool over the 4 pyramid stages
        desc_dim = dpt_feats * 4

        # ----- dense segmentation head (trainable): 96x96 dense feature -> 1-ch logit -> upsample to crop
        self.seg_out_size = (self.img_size, self.img_size)   # per-pixel logits at the crop resolution
        self.seg_head = nn.Sequential(
            nn.Conv2d(dpt_feats, dpt_feats, 3, 1, 1, bias=False), nn.BatchNorm2d(dpt_feats), nn.ReLU(inplace=True),
            nn.Conv2d(dpt_feats, dpt_feats // 2, 3, 1, 1, bias=False), nn.BatchNorm2d(dpt_feats // 2), nn.ReLU(inplace=True),
            nn.Conv2d(dpt_feats // 2, dpt_feats // 2, 3, 1, 1, bias=False), nn.BatchNorm2d(dpt_feats // 2), nn.ReLU(inplace=True),
            nn.Conv2d(dpt_feats // 2, 1, 1),   # binary hand-object foreground logit
        )

        # ----- contact / hand-contact heads (trainable) -----
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
        # ----- fingertip-force head (trainable): n_force=6 dims -----------------------------------
        #   [0..4] = thumb, index, middle, ring, pinky fingertips ; [5] = palm   (FEEL sensor order)
        # Supervised ONLY by force datasets (FEEL); the 12 contact datasets mask this term out, and
        # FEEL in turn masks out the contact-map/binary terms. Regressed in LOG1P newtons -- raw force
        # is heavily right-skewed (mean ~0.05 N, max ~10 N) so log1p compresses the range and keeps
        # the loss from being dominated by rare spikes. Decode with expm1(x).clamp(min=0).
        # n_force=0 disables the head entirely, which keeps pre-force checkpoints loadable STRICT.
        self.n_force = int(n_force)
        self.force_head = nn.Sequential(
            nn.Linear(desc_dim, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, self.n_force),
        ) if self.n_force > 0 else None
        # ----- hurdle GATE: P(this fingertip is loaded) ------------------------------------------
        # A single regressor on FEEL's 75%-zero target is minimised by predicting ~0, so the force
        # task is split two-part (see lib/core/force_utils.py): this gate trains on EVERY reading,
        # while force_head trains only on loaded ones. Reported force is gate * magnitude -- the
        # magnitude head alone would emit force on a resting hand, having never seen a zero.
        self.force_hurdle = bool(force_hurdle) and self.n_force > 0
        self.force_presence_head = nn.Sequential(
            nn.Linear(desc_dim, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, self.n_force),
        ) if self.force_hurdle else None
        self._init_heads()

    def _init_heads(self):
        heads = list(self.contact_head) + list(self.hand_contact_head)
        if getattr(self, 'force_head', None) is not None:
            heads = heads + list(self.force_head)
        if getattr(self, 'force_presence_head', None) is not None:
            heads = heads + list(self.force_presence_head)
        for m in heads:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    # ---------- checkpoint compatibility ----------
    def load_state_dict(self, state_dict, strict=True, **kwargs):
        """Reshape the force head to match the checkpoint, so STRICT loads keep working both ways.

        The 6-dim force head postdates the 12-dataset contact checkpoints, so those have no
        `force_head.*` keys. Rather than making every inference script aware of that, we adapt SELF
        to the checkpoint before delegating -- but ONLY when the caller asked for a strict load:

          strict=True , ckpt has no force head -> drop ours   (e.g. running vitl_12ds_epoch9 in the
                                                               meshtop/Stage-B pipeline, which
                                                               asserts a STRICT load)
          strict=True , ckpt has a force head  -> build ours from the checkpoint's output dim
          strict=False                         -> leave the model alone. This is the WARM-START path
                                                  (`finetune_*.py` loads a contact checkpoint with
                                                  strict=False to start force training from it), and
                                                  there the freshly-initialised force head must
                                                  survive as-is.
        """
        if strict:
            # the hurdle gate is optional and independent of the magnitude head, so match it first
            ck_pres = any(k.startswith('force_presence_head.') for k in state_dict)
            have_pres = getattr(self, 'force_presence_head', None) is not None
            if not ck_pres and have_pres:
                print("[HACO_VJEPA] checkpoint has no hurdle gate -> dropping force_presence_head "
                      "(pre-hurdle force checkpoint)", flush=True)
                self.force_presence_head = None
                self.force_hurdle = False
            elif ck_pres and not have_pres:
                w = sorted((k for k in state_dict if k.startswith('force_presence_head.')
                            and k.endswith('.weight')), key=lambda k: int(k.split('.')[1]))
                n_out = int(state_dict[w[-1]].shape[0])
                desc_dim = int(self.desc_norm.normalized_shape[0])
                print(f"[HACO_VJEPA] checkpoint carries a hurdle gate -> building "
                      f"force_presence_head (n_force={n_out})", flush=True)
                dev = next(self.parameters()).device
                self.force_presence_head = nn.Sequential(
                    nn.Linear(desc_dim, 512), nn.GELU(), nn.Dropout(0.1),
                    nn.Linear(512, 256), nn.GELU(),
                    nn.Linear(256, n_out),
                ).to(dev)
                self.force_hurdle = True
            ck_force = sorted((k for k in state_dict if k.startswith('force_head.')),
                              key=lambda k: int(k.split('.')[1]))
            have = getattr(self, 'force_head', None) is not None
            if not ck_force and have:
                print(f"[HACO_VJEPA] checkpoint predates the force head -> dropping force_head "
                      f"(n_force {self.n_force} -> 0)", flush=True)
                self.force_head = None
                self.n_force = 0
                self.force_presence_head = None
                self.force_hurdle = False
            elif ck_force and not have:
                w = [k for k in ck_force if k.endswith('.weight')]
                n_out = int(state_dict[w[-1]].shape[0])
                desc_dim = int(self.desc_norm.normalized_shape[0])
                print(f"[HACO_VJEPA] checkpoint carries a force head -> building it (n_force={n_out})",
                      flush=True)
                dev = next(self.parameters()).device
                self.force_head = nn.Sequential(
                    nn.Linear(desc_dim, 512), nn.GELU(), nn.Dropout(0.1),
                    nn.Linear(512, 256), nn.GELU(),
                    nn.Linear(256, n_out),
                ).to(dev)
                self.n_force = n_out
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    # ---------- freeze / unfreeze control ----------
    def set_backbone_trainable(self, trainable):
        """Flip the backbone between frozen (eval, no grad) and trainable (train, grad-checkpointed)."""
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
        # the V-JEPA encoder checkpoints each block when this attribute is set
        self.backbone.use_activation_checkpointing = bool(flag)

    def set_surgical_finetune(self, top_k):
        """SURGICAL fine-tune: freeze patch_embed + the bottom (n_layers - top_k) blocks; train ONLY
        the top_k blocks (+ their trainable-region tap norms), the DPT decoder, and both heads.

        The frozen bottom blocks run under torch.no_grad() in _encode_surgical (inference, detached),
        so they build NO autograd graph and need NO gradient-checkpoint recompute -- only the top_k
        blocks + head build the graph. This is the whole point: far less backward compute + memory.
        """
        n = self.n_layers
        top_k = int(top_k)
        assert 1 <= top_k <= n, f"top_k must be in [1,{n}]"
        split = n - top_k
        self._surgical = True
        self._surgical_split = split
        self.freeze_backbone = False                 # trainable blocks train; frozen ones via no_grad
        self.enable_gradient_checkpointing(False)    # custom forward; encoder-level ckpt unused
        # freeze the whole backbone, then unfreeze the top_k blocks
        self.backbone.requires_grad_(False)
        for i in range(split, n):
            self.backbone.blocks[i].requires_grad_(True)
        # tap norms: only those in the TRAINABLE region (layer index >= split) can receive grad
        for oi, L in enumerate(self.hook_layers):
            self.backbone.norms_block[oi].requires_grad_(int(L) >= split)
        self.backbone.train()                        # LayerNorm-only ViT: train()==eval() numerically
        return self

    def _encode_surgical(self, video):
        """Forward the encoder with the bottom (split) blocks under no_grad and the top blocks with
        grad; collect the 4 tapped target-frame maps. Mirrors VisionTransformer.forward's video path."""
        be = self.backbone
        _, _, T, H, W = video.shape
        Tp = T // be.tubelet_size
        Hp, Wp = H // be.patch_size, W // be.patch_size
        taps = [None] * len(self.hook_layers)
        split = self._surgical_split
        with torch.no_grad():                        # frozen bottom blocks: no autograd graph built
            x = be.patch_embed(video)
            if getattr(be, 'modality_embedding', False):
                x = x + be.video_mod_embed.repeat(x.shape[0], 1, 1)
            for i in range(split):
                x, _ = be.blocks[i](x, mask=None, T=Tp, H_patches=Hp, W_patches=Wp,
                                    return_attn=False, mode='video')
                if i in self.hook_layers:
                    oi = self.hook_layers.index(i)
                    taps[oi] = be.norms_block[oi](x)
        x = x.detach()                               # cut the graph at the split
        for i in range(split, be.get_num_layers()):
            x, _ = be.blocks[i](x, mask=None, T=Tp, H_patches=Hp, W_patches=Wp,
                                return_attn=False, mode='video')
            if i in self.hook_layers:
                oi = self.hook_layers.index(i)
                taps[oi] = be.norms_block[oi](x)
        return [self._last_tubelet_map(t.float()) for t in taps]

    def train(self, mode=True):
        """Keep the frozen backbone in eval() even when the whole model is set to train()."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    # ---------- token -> target-frame spatial map ----------
    def _last_tubelet_map(self, tokens):
        """[B, N, hidden] -> [B, hidden, grid, grid] for the LAST temporal tubelet (target frame)."""
        B, N, C = tokens.shape
        s = self.grid * self.grid                 # spatial tokens per temporal slice (576)
        tub = tokens[:, N - s:N, :]               # last temporal slice -> target frame
        return tub.transpose(1, 2).reshape(B, C, self.grid, self.grid)

    def _encode(self, clip):
        """Run the V-JEPA 2.1 encoder -> 4 hooked target-frame maps [B, hidden, grid, grid]."""
        # loader emits [B, T, 3, H, W]; the encoder wants [B, 3, T, H, W]
        video = clip.permute(0, 2, 1, 3, 4).contiguous()
        if self._surgical:
            return self._encode_surgical(video)
        if self.freeze_backbone:
            with torch.no_grad():
                outs = self.backbone(video)          # list of 4 [B, N, hidden]
            outs = [o.detach() for o in outs]
        else:
            outs = self.backbone(video)
        return [self._last_tubelet_map(o.float()) for o in outs]

    # ---------- forward ----------
    def forward(self, inputs, mode="test"):
        clip = inputs["input"]["image"].to(self.device)   # [B, T, 3, 384, 384]
        maps = self._encode(clip)
        pyramid = self.decoder(maps)                        # [p3@24, p2@48, p1@96, dense@96]
        # multi-scale context descriptor: global-avg-pool each stage, concat
        desc = torch.cat([p.mean(dim=(2, 3)) for p in pyramid], dim=1)   # [B, feats*4]
        desc = self.desc_norm(desc)

        contact_out = self.contact_head(desc)               # [B, 778]
        hand_contact_out = self.hand_contact_head(desc)     # [B, binary_out]

        # dense segmentation: finest pyramid feature (dense@96) -> 1-ch logit -> upsample to crop res
        seg_logits = self.seg_head(pyramid[-1])             # [B, 1, 96, 96]
        seg_out = F.interpolate(seg_logits, size=self.seg_out_size, mode="bilinear", align_corners=False)
        out = dict(contact_out=contact_out, hand_contact_out=hand_contact_out, seg_out=seg_out)
        if self.force_head is not None:
            out['force_out'] = self.force_head(desc)          # [B, n_force] log1p(scaled force)
        if self.force_presence_head is not None:
            out['force_logit'] = self.force_presence_head(desc)   # [B, n_force] P(loaded) logits
        return out

    # ---------- introspection ----------
    def trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def frozen_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)
