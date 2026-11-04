"""
HACO_VJEPA: frozen V-JEPA 2 (video) backbone + trainable DPT decoder + dual contact head.

Backbone
  A transformers `VJEPA2Model` chosen by `model_id`; the released checkpoint uses
  facebook/vjepa2-vith-fpc64-256 (ViT-H, hidden=1280, 32 layers, patch=16, tubelet=2, 256x256,
  RoPE position encoding). Hidden size and depth are read from the model config. FROZEN by default
  (requires_grad_(False), .eval(), forward under torch.no_grad()) so no backbone activations
  are stored for autograd. `freeze_backbone=False` flips it to trainable (for later experiments).

Input / tokens
  A clip of T RGB frames at 256x256 (T=8 for the released checkpoint): pixel_values_videos =
  [B, T, 3, 256, 256]. The Conv3d patch-embed (tubelet=2) yields T'=T/2 temporal x 16x16 spatial tokens, ordered
  TEMPORAL-MAJOR (token index = t'*256 + h*16 + w; verified from VJEPA2PatchEmbeddings3D:
  proj(x).flatten(2).transpose(1,2)). RoPE handles T=16 natively, so NO temporal positional-
  embedding interpolation is needed (there is no learned absolute temporal pos-emb).

Target-frame feature
  The model predicts contact for the LAST (target) frame of the clip. We take the LAST temporal
  tubelet's tokens -- indices [ (T'-1)*256 : T'*256 ] = [1792:2048] -- which is the tubelet that
  covers the final two frames (the target). Those 256 tokens reshape to a 16x16 grid -> a
  [B, 1024, 16, 16] spatial feature map for the target frame. (Documented choice: "last temporal
  tubelet" rather than time-pooling, because the target frame lives in that last tubelet.)

DPT decoder (TRAINABLE)
  We hook 4 intermediate transformer layers (output_hidden_states gives 25 states: embeddings +
  24 layers; we use states [6, 12, 18, 24], DPT-style evenly spaced). For each we extract the
  target-frame 16x16x1024 map, project to a common width, "reassemble" to 4 resolutions
  (32x32, 16x16, 8x8, 4x4 via transpose-conv / identity / strided convs), then fuse top-down
  with RefineNet-style feature-fusion blocks into a single dense [B, F, 32, 32] feature.

Heads (match the existing HACO output dict so losses / eval are reusable)
  * contact-map head: global-pool the dense feature -> MLP -> 778 logits  -> contact_out  [B,778]
  * binary head:      global-pool the dense feature -> MLP -> 2  logits  -> hand_contact_out [B,2]
    (binary head width is a ctor flag, default 2 = [left, right]).

forward(inputs, mode) -> dict(contact_out=[B,778], hand_contact_out=[B,2]).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


VJEPA2_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"


# ------------------------------------------------------------------ DPT building blocks
class _ResidualConvUnit(nn.Module):
    def __init__(self, feats):
        super().__init__()
        self.conv1 = nn.Conv2d(feats, feats, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(feats)
        self.conv2 = nn.Conv2d(feats, feats, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(feats)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(self.act(x))
        out = self.bn1(out)
        out = self.conv2(self.act(out))
        out = self.bn2(out)
        return out + x


class _FeatureFusionBlock(nn.Module):
    """RefineNet-style fusion: (optionally add a skip), refine, upsample x2."""
    def __init__(self, feats):
        super().__init__()
        self.rcu1 = _ResidualConvUnit(feats)
        self.rcu2 = _ResidualConvUnit(feats)
        self.out_conv = nn.Conv2d(feats, feats, 1, 1, 0)

    def forward(self, x, skip=None):
        if skip is not None:
            x = x + self.rcu1(skip)
        x = self.rcu2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.out_conv(x)
        return x


class DPTDecoder(nn.Module):
    """DPT-style reassemble + multi-scale fusion over 4 hooked ViT layers.

    Inputs: list of 4 spatial maps [B, hidden, 16, 16] (shallow -> deep).
    Output: dense feature [B, feats, 32, 32].
    """
    def __init__(self, hidden=1024, feats=256, readout_chans=(256, 256, 256, 256)):
        super().__init__()
        self.feats = feats
        # 1x1 projections from ViT hidden dim -> per-stage channels
        self.proj = nn.ModuleList([nn.Conv2d(hidden, c, 1, 1, 0) for c in readout_chans])
        # resample each stage to a distinct resolution (DPT reassemble):
        #   stage0 16->32 (transpose conv x2), stage1 16->16 (identity conv),
        #   stage2 16->8  (stride-2 conv),       stage3 16->4 (stride-4 conv)
        self.resample = nn.ModuleList([
            nn.ConvTranspose2d(readout_chans[0], readout_chans[0], 2, 2, 0),
            nn.Conv2d(readout_chans[1], readout_chans[1], 3, 1, 1),
            nn.Conv2d(readout_chans[2], readout_chans[2], 3, 2, 1),
            nn.Conv2d(readout_chans[3], readout_chans[3], 3, 4, 1),
        ])
        # unify to `feats` channels
        self.to_feats = nn.ModuleList([nn.Conv2d(c, feats, 3, 1, 1, bias=False) for c in readout_chans])
        # fusion blocks, deep -> shallow
        self.fuse3 = _FeatureFusionBlock(feats)  # 4  -> 8
        self.fuse2 = _FeatureFusionBlock(feats)  # 8  -> 16
        self.fuse1 = _FeatureFusionBlock(feats)  # 16 -> 32
        self.fuse0 = _FeatureFusionBlock(feats)  # 32 -> 64  (then we pool)
        self.out_conv = nn.Sequential(
            nn.Conv2d(feats, feats, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feats),
            nn.ReLU(inplace=True),
        )

    def forward(self, maps):
        # maps[i]: [B, hidden, 16, 16]
        f = [self.to_feats[i](self.resample[i](self.proj[i](m))) for i, m in enumerate(maps)]
        # f[0]:32x32  f[1]:16x16  f[2]:8x8  f[3]:4x4
        x = self.fuse3(f[3])            # 4 -> 8
        x = self.fuse2(x, skip=f[2])    # 8 -> 16
        x = self.fuse1(x, skip=f[1])    # 16 -> 32
        # bring f[0] (32x32) in as the shallowest skip, fuse0 upsamples 32->64
        x = self.fuse0(x, skip=f[0])    # 32 -> 64
        x = self.out_conv(x)
        return x  # [B, feats, 64, 64]


# ------------------------------------------------------------------ full model
class HACO_VJEPA(nn.Module):
    def __init__(self, freeze_backbone=True, binary_out=2, hook_layers=(6, 12, 18, 24),
                 dpt_feats=256, model_id=VJEPA2_MODEL_ID, torch_dtype=torch.float32):
        super().__init__()
        from transformers import VJEPA2Model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.freeze_backbone = freeze_backbone
        self.binary_out = binary_out
        self.hook_layers = tuple(hook_layers)   # indices into hidden_states (0=embeddings .. 24)
        self.model_id = model_id

        # ----- backbone -----
        self.backbone = VJEPA2Model.from_pretrained(model_id, torch_dtype=torch_dtype)
        # We only use the encoder (we read encoder hidden states). The predictor is never used;
        # drop it so its params don't sit in the module (saves memory + avoids dangling unused
        # parameters under DDP).
        if hasattr(self.backbone, 'predictor') and self.backbone.predictor is not None:
            self.backbone.predictor = None
        self.hidden = int(self.backbone.config.hidden_size)        # 1024
        self.tubelet = int(self.backbone.config.tubelet_size)      # 2
        self.patch = int(self.backbone.config.patch_size)          # 16
        self.grid = cfg_input = 256 // self.patch                  # 16 tokens per side
        self.n_layers = int(self.backbone.config.num_hidden_layers)
        assert all(0 <= h <= self.n_layers for h in self.hook_layers)

        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        # ----- DPT decoder (trainable) -----
        self.decoder = DPTDecoder(hidden=self.hidden, feats=dpt_feats)

        # ----- heads (trainable) -----
        self.contact_head = nn.Sequential(
            nn.Linear(dpt_feats, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, 778),
        )
        self.hand_contact_head = nn.Sequential(
            nn.Linear(dpt_feats, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, self.binary_out),
        )
        self._init_heads()

    def _init_heads(self):
        for m in list(self.contact_head) + list(self.hand_contact_head):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def train(self, mode=True):
        """Keep the frozen backbone in eval() even when the model is set to train()."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    # ---------- token -> target-frame spatial map ----------
    def _last_tubelet_map(self, hidden_state):
        """[B, 2048, hidden] -> [B, hidden, 16, 16] for the LAST temporal tubelet."""
        B, N, C = hidden_state.shape
        s = self.grid * self.grid                  # 256 spatial tokens per temporal slice
        tubelet_tokens = hidden_state[:, N - s:N, :]   # last temporal slice -> target frame
        x = tubelet_tokens.transpose(1, 2).reshape(B, C, self.grid, self.grid)
        return x

    def _encode(self, clip):
        """Run the (frozen) V-JEPA encoder and return the 4 hooked target-frame maps.

        We call the encoder directly (the top-level VJEPA2Model.forward also runs the unused
        predictor; we dropped it). The encoder returns BaseModelOutput with `hidden_states` =
        (embeddings, layer1, ..., layer24) when output_hidden_states=True.
        """
        if self.freeze_backbone:
            with torch.no_grad():
                out = self.backbone.encoder(pixel_values_videos=clip, output_hidden_states=True)
            hs = [out.hidden_states[i].detach() for i in self.hook_layers]
        else:
            out = self.backbone.encoder(pixel_values_videos=clip, output_hidden_states=True)
            hs = [out.hidden_states[i] for i in self.hook_layers]
        maps = [self._last_tubelet_map(h.float()) for h in hs]
        return maps

    # ---------- forward ----------
    def forward(self, inputs, mode='test'):
        clip = inputs['input']['image'].to(self.device)   # [B, 16, 3, 256, 256]
        maps = self._encode(clip)
        dense = self.decoder(maps)                          # [B, feats, 64, 64]
        pooled = dense.mean(dim=(2, 3))                     # [B, feats]

        contact_out = self.contact_head(pooled)             # [B, 778]
        hand_contact_out = self.hand_contact_head(pooled)   # [B, binary_out]

        return dict(contact_out=contact_out, hand_contact_out=hand_contact_out)

    # ---------- introspection ----------
    def trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def frozen_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)
