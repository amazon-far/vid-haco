"""
Training objective for both Vid-HACO models.

    total = contact_map_loss + binary_loss + seg_loss

contact_map_loss (778 logits; the objective of HACO, Jung & Lee, NeurIPS 2025, restricted to the 778-vertex
map — the only contact output these models have):
    vcb + 0.1 * reg + smooth
  vcb     vertex-level class-balanced BCE. Each vertex gets (no-contact, contact) class weights from the
          "effective number of samples" rule (Cui et al., CVPR 2019) applied to label counts taken from
          HACO's training-set contact statistics; the counts blend from global to per-vertex over training
          (see `_class_counts`). Plus 0.5 x the mean predicted probability on non-contact vertices.
  reg     L1 between the predicted probabilities and the per-vertex contact frequency.
  smooth  isolation penalty: contact (and non-contact) probability that disagrees with its 1-ring
          neighbourhood on the MANO mesh is penalised, normalised by the propagated mass.
binary_loss  BCE-with-logits on the 2-bit [left, right] hand-contact head.
seg_loss     BCE + soft Dice on the ViT-L model's dense output, only for samples with has_seg. No loader
             supplies a mask (data_processing/multidata_vjepa.py), so this is a graph-connected zero for
             every batch of the released runs; it is kept so that every parameter receives a gradient on
             every step, which DistributedDataParallel(static_graph=True) requires.

Inputs read from disk (HACO's base data release, not redistributed here):
  data/base_data/contact_data/all/contact_data_all.npy   [N, 778] 0/1 labels -> class counts
  data/base_data/contact_data/all/contact_means_all.npy  [778]   per-vertex contact frequency
and the MANO right-hand faces from inference/data/base_data/mano_assets.npz (tools/make_mano_assets.py).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CONTACT_DATA = 'data/base_data/contact_data/all/contact_data_all.npy'
CONTACT_MEANS = 'data/base_data/contact_data/all/contact_means_all.npy'

VCB_BETA = 0.9999
VCB_FP_WEIGHT = 0.5         # weight of the mean-probability-on-non-contact term inside vcb
VCB_PER_VERTEX_MAX = 0.3    # final share of the per-vertex counts in the blended class counts
VCB_COUNT_SCALE = 1000.0    # counts are rescaled to sum to this per vertex
VCB_COUNT_FLOOR = 20.0      # ... after flooring each class at this value
REG_WEIGHT = 0.1


def _label_counts(path, chunk=65536):
    """Per-vertex (no-contact, contact) label counts from an [N, 778] 0/1 .npy, streamed from a memory
    map so that no rank holds the ~5 GB array (or a same-size temporary) in RAM. -> int64 [778, 2]."""
    labels = np.load(path, mmap_mode='r')
    counts = np.zeros((labels.shape[1], 2), np.int64)
    for i in range(0, labels.shape[0], chunk):
        block = np.asarray(labels[i:i + chunk])
        counts[:, 0] += (block == 0).sum(0)
        counts[:, 1] += (block == 1).sum(0)
    return counts


def _class_counts(counts, total_epochs):
    """Per-epoch (no-contact, contact) counts for every vertex: [total_epochs, 778, 2].

    counts: int64 [778, 2] label counts. Global counts (all vertices pooled) and per-vertex counts are
    each normalised to sum to VCB_COUNT_SCALE, then blended with a per-vertex share that ramps linearly
    from 0 at the first epoch to VCB_PER_VERTEX_MAX at the last. Each class is floored at
    VCB_COUNT_FLOOR and the pair rescaled to VCB_COUNT_SCALE again.
    """
    if total_epochs < 2:
        raise ValueError("the VCB weight ramp needs total_epochs >= 2")
    counts = np.asarray(counts, np.int64)
    per_vertex = torch.as_tensor(counts).float()
    per_vertex = per_vertex / per_vertex.sum(1, keepdim=True).clamp(min=1e-6) * VCB_COUNT_SCALE
    pooled = torch.as_tensor(counts.sum(0)).float()          # exact integer totals, then fp32
    pooled = pooled / pooled.sum().clamp(min=1e-6) * VCB_COUNT_SCALE

    out = []
    for e in range(total_epochs):
        w = VCB_PER_VERTEX_MAX * e / (total_epochs - 1)
        c = (w * per_vertex + (1.0 - w) * pooled).clamp(min=VCB_COUNT_FLOOR)
        out.append(c / c.sum(1, keepdim=True) * VCB_COUNT_SCALE)
    return torch.stack(out)


def _effective_number_weights(counts, beta=VCB_BETA):
    """Class weights proportional to 1 / effective number (1 - beta^n) / (1 - beta), normalised per
    vertex to sum to 1. counts: [778, 2] -> [778, 2]."""
    w = (1.0 - beta) / (1.0 - torch.pow(beta, counts))
    return w / w.sum(1, keepdim=True)


def _mesh_adjacency(faces, num_verts=778):
    """Dense 1-ring adjacency with self-loops. Each face adds both directions of its three edges, so an
    edge shared by two faces has weight 2; the diagonal is 1."""
    f = torch.as_tensor(np.asarray(faces), dtype=torch.long)
    src = torch.cat([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    dst = torch.cat([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    adj = torch.zeros(num_verts, num_verts)
    adj.index_put_((src, dst), torch.ones(src.numel()), accumulate=True)
    return adj + torch.eye(num_verts)


def _seg_loss(seg_logits, seg_gt, has_seg):
    """BCE + soft Dice per sample, averaged over samples with a mask; graph-connected zero otherwise."""
    logits = seg_logits[:, 0].float()
    m = has_seg.float().view(-1)
    if float(m.sum()) < 1.0:
        return logits.sum() * 0.0
    gt = seg_gt.float()
    bce = F.binary_cross_entropy_with_logits(logits, gt, reduction='none').mean(dim=(1, 2))
    p = torch.sigmoid(logits)
    dice = 1.0 - (2.0 * (p * gt).sum(dim=(1, 2)) + 1.0) / (p.sum(dim=(1, 2)) + gt.sum(dim=(1, 2)) + 1.0)
    return ((bce + dice) * m).sum() / m.sum()


class ContactLoss(nn.Module):
    def __init__(self, faces, total_epochs, contact_data=CONTACT_DATA, contact_means=CONTACT_MEANS):
        super().__init__()
        counts = _class_counts(_label_counts(contact_data), total_epochs)
        weights = torch.stack([_effective_number_weights(c) for c in counts])    # [E, 778, 2]
        self.register_buffer('class_weights', weights, persistent=False)
        means = torch.as_tensor(np.load(contact_means), dtype=torch.float32)
        self.register_buffer('contact_means', means.view(1, -1), persistent=False)
        self.register_buffer('adjacency', _mesh_adjacency(faces), persistent=False)

    # ---- contact-map terms (all on probabilities, in fp32) ----
    def vcb(self, logits, gt, epoch):
        w = self.class_weights[min(int(epoch), self.class_weights.shape[0] - 1)]   # [778, 2]
        prob = logits.sigmoid()
        w = w[torch.arange(gt.shape[1], device=gt.device)[None, :], gt.long()]     # weight of the GT class
        bce = F.binary_cross_entropy(prob, gt, reduction='none')
        return (bce * w).mean() + VCB_FP_WEIGHT * ((gt == 0).float() * prob).mean()

    def reg(self, logits):
        return F.l1_loss(logits.sigmoid(), self.contact_means.expand(logits.shape[0], -1))

    def smooth(self, logits):
        p = logits.sigmoid()
        q = 1.0 - p
        prop_p = p @ self.adjacency       # adjacency is symmetric
        prop_q = q @ self.adjacency
        iso = (p - prop_p / (prop_p.max(1, keepdim=True)[0] + 1e-6)).abs().sum(1) \
            + (q - prop_q / (prop_q.max(1, keepdim=True)[0] + 1e-6)).abs().sum(1)
        mass = prop_p.sum(1) + prop_q.sum(1) + 1e-3
        return torch.log1p(iso / mass).mean()

    def forward(self, preds, targets, epoch, has_seg=None):
        """preds: model output dict. targets: the loader's `targets_data`. has_seg: the loader's
        `meta_info.has_seg` ([B] bool); None means no sample has a mask."""
        logits = preds['contact_out'].float()
        dev = logits.device
        gt = targets['contact_data']['contact_h'].to(dev).float()

        vcb = self.vcb(logits, gt, epoch)
        reg = self.reg(logits)
        smooth = self.smooth(logits)
        contact_map_loss = vcb + REG_WEIGHT * reg + smooth

        hand_logits = preds['hand_contact_out'].float()
        hand_gt = targets['hand_contact_gt'].to(dev).float()
        binary_loss = F.binary_cross_entropy_with_logits(hand_logits, hand_gt)

        if 'seg_out' in preds:
            if has_seg is None:
                has_seg = torch.zeros(logits.shape[0], dtype=torch.bool)
            seg_loss = _seg_loss(preds['seg_out'], targets['seg_mask'].to(dev), has_seg.to(dev))
        else:
            seg_loss = logits.new_zeros(())

        total = contact_map_loss + binary_loss + seg_loss
        with torch.no_grad():
            binary_acc = ((hand_logits.sigmoid() > 0.5).float() == hand_gt).float().mean()
        return total, dict(total_loss=total, contact_map_loss=contact_map_loss, vcb_loss=vcb,
                           reg_loss=reg, smooth_loss=smooth, binary_loss=binary_loss,
                           seg_loss=seg_loss.detach(), binary_acc=binary_acc)
