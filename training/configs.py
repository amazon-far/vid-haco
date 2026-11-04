"""
The two training recipes behind the released checkpoints.

Every value below was read back from the `args` / `optimizer` / `scheduler` state saved inside the
original run's own epoch checkpoint, not from memory or a launch script:

  vith  experiments_vjepa_h_dpt_8f_seq/checkpoints/haco_vjepa_alldata_epoch1.ckpt
        -> released as vith.pt (epoch 1, val contact-map F1 0.637, binary-contact F1 0.987)
  vitl  checkpoints_vitl_12ds/haco_vjepa_alldata_epoch6.ckpt   (exp_dir experiments_vjepa_l_dpt_8f_seq_12ds)
        -> released as vitl.pt (epoch 6, val contact-map F1 0.791, binary-contact F1 0.985)

World size is not stored in the checkpoint; it follows from the step counter: vith reached
global_step 13,982 after 2 epochs of a 447,471-clip train split at 8 clips/GPU -> 6,991 steps/epoch
= 447,471 // 64 -> 8 GPUs. vitl reached 28,315 after 7 epochs of 1,553,353 clips at 12 clips/GPU ->
4,045 steps/epoch = 1,553,353 // 384 -> 32 GPUs (4 nodes x 8).

Anything a recipe does not set falls back to train.py's argparse default.
"""

# Dataset rosters, in the order the original runs listed them. The order fixes each dataset's id in the
# per-dataset validation table and the layout of the concatenated train set the sampler shuffles.
VITH_DATASETS = ['HOI4D', 'ARCTIC', 'HIC', 'MOW', 'H2O', 'PROX', 'InterHand26M', 'Decaf', 'Hi4D']
VITL_DATASETS = ['HOI4D', 'ARCTIC', 'ARCTIC_allo', 'InterHand26M', 'HIC', 'MOW', 'H2O', 'PROX', 'Decaf',
                 'Hi4D', 'epic_contact', 'HOT3D']

RECIPES = {
    # V-JEPA 2 ViT-H/16 at 256 px (transformers VJEPA2Model) + DPT decoder; 9 datasets.
    'vith': dict(
        model_id='facebook/vjepa2-vith-fpc64-256',
        input_size=256,
        datasets=VITH_DATASETS,
        frames=8, stride=2,
        batch=8,                 # clips per GPU; x 8 GPUs = 64 per step
        epochs=10,               # planned length; also sets the VCB loss's vertex-weight ramp
        lr=1e-4,                 # DPT decoder + heads
        backbone_lr=1e-5,        # V-JEPA encoder
        lr_schedule='cosine',    # linear warmup (500 steps), then cosine to 0 over all `epochs`
        world_size=8,
        released_epoch=1,
    ),
    # V-JEPA 2.1 ViT-L/16 at 384 px (vjepa2 repo builder) + DPT decoder + seg head; 12 datasets.
    'vitl': dict(
        model_id='vjepa2.1-vitl-384',
        input_size=384,
        datasets=VITL_DATASETS,
        frames=8, stride=2,
        batch=12,                # clips per GPU; x 32 GPUs = 384 per step
        epochs=10,
        lr=1e-4,
        backbone_lr=5e-6,
        lr_schedule='warmconst',  # linear warmup (500 steps), then constant
        world_size=32,
        released_epoch=6,
    ),
}

# Shared by both runs (same values in both checkpoints).
COMMON = dict(
    weight_decay=1e-4,       # AdamW, betas (0.9, 0.95)
    betas=(0.9, 0.95),
    amp='bf16',
    thres=0.52,              # contact threshold for the validation P/R/F1
    seed=314,                # + rank, as in the original trainer
    resize_factor=0.5,       # HOI4D frames are read from the half-resolution tree
)
