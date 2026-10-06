"""Small shared-backbone models for a controlled segmentation pilot."""
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(4, cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(4, cout), nn.SiLU(),
        )


class TumorNet(nn.Module):
    """baseline, softroi ablation, or srb (SoftROI-Boundary U-Net).

    Only MRI intensities enter forward(). Masks supervise training losses only.
    A soft ROI reweights skip features while retaining a 0.5 context floor.
    The full model predicts a boundary map and learns a residual correction,
    weighted by segmentation uncertainty 4*p*(1-p).
    """
    def __init__(self, variant="baseline", width=16):
        super().__init__()
        if variant not in {"baseline", "softroi", "srb"}:
            raise ValueError(variant)
        self.variant = variant
        c = [width * 2**i for i in range(4)]
        self.encoders = nn.ModuleList([Block(1, c[0])] + [Block(c[i-1], c[i]) for i in range(1, 4)])
        self.decoders = nn.ModuleList([Block(c[i+1] + c[i], c[i]) for i in [2, 1, 0]])
        self.seg_head = nn.Conv2d(c[0], 1, 1)
        self.class_head = nn.Linear(c[-1], 3)
        if variant != "baseline":
            self.coarse_head = nn.Conv2d(c[-1], 1, 1)
        if variant == "srb":
            self.boundary_head = nn.Sequential(nn.Conv2d(c[0], 8, 3, padding=1), nn.SiLU(), nn.Conv2d(8, 1, 1))
            self.refiner = nn.Sequential(nn.Conv2d(c[0]+3, c[0], 3, padding=1), nn.SiLU(), nn.Conv2d(c[0], 1, 1))
            nn.init.zeros_(self.refiner[-1].weight)
            nn.init.zeros_(self.refiner[-1].bias)

    def forward(self, x):
        features = []
        for i, enc in enumerate(self.encoders):
            x = enc(F.max_pool2d(x, 2) if i else x)
            features.append(x)
        out = {"class_logits": self.class_head(x.mean((2, 3)))}
        coarse = self.coarse_head(x) if self.variant != "baseline" else None
        if coarse is not None:
            out["coarse_logits"] = coarse
        for decoder, skip in zip(self.decoders, reversed(features[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            if coarse is not None:
                gate = 0.5 + F.interpolate(coarse.sigmoid(), size=skip.shape[-2:], mode="bilinear", align_corners=False)
                skip = skip * gate
            x = decoder(torch.cat([x, skip], 1))
        base = self.seg_head(x)
        out["base_logits"] = base
        if self.variant == "srb":
            edge = self.boundary_head(x)
            roi = F.interpolate(coarse.sigmoid(), size=x.shape[-2:], mode="bilinear", align_corners=False)
            uncertainty = 4 * base.sigmoid() * (1 - base.sigmoid())
            correction = self.refiner(torch.cat([x, roi, edge.sigmoid(), uncertainty], 1))
            out["boundary_logits"] = edge
            out["seg_logits"] = base + uncertainty * correction
        else:
            out["seg_logits"] = base
        return out


def segmentation_loss(logits, mask, positive_weight=3.0):
    bce = F.binary_cross_entropy_with_logits(logits, mask, pos_weight=logits.new_tensor(positive_weight))
    p = logits.sigmoid()
    dice = (2 * (p * mask).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + mask.sum((1, 2, 3)) + 1)
    return 0.5 * bce + 0.5 * (1 - dice.mean())


def boundary_target(mask):
    dilation = F.max_pool2d(mask, 3, stride=1, padding=1)
    erosion = -F.max_pool2d(-mask, 3, stride=1, padding=1)
    return dilation - erosion


def objective(out, mask, labels):
    loss = segmentation_loss(out["seg_logits"], mask)
    loss = loss + 0.15 * F.cross_entropy(out["class_logits"], labels)
    if "coarse_logits" in out:
        # Max pooling preserves small lesions as foreground at coarse resolution.
        coarse_mask = F.adaptive_max_pool2d(mask, out["coarse_logits"].shape[-2:])
        loss = loss + 0.25 * segmentation_loss(out["coarse_logits"], coarse_mask)
    if "boundary_logits" in out:
        loss = loss + 0.10 * segmentation_loss(out["boundary_logits"], boundary_target(mask), 8.0)
    return loss
