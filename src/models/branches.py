"""Each branch = backbone + KAN downsampler, trained independently in Step 4,
then reused (encoder frozen initially) inside the fusion model in Step 5."""
import torch
import torch.nn as nn

from .backbone import MobileNetV3Backbone
from .kan import KANHead


class BranchModel(nn.Module):
    """Generic single-branch model: backbone -> KAN downsample -> KAN scalar head.
    Used standalone during Step 4's independent branch training/validation, and its
    `.backbone` + `.downsample` are what get plugged into the fusion model afterward."""

    def __init__(self, cfg: dict, force_mlp_fallback: bool = False):
        super().__init__()
        self.backbone = MobileNetV3Backbone(cfg["model"]["backbone"], pretrained=cfg["model"]["pretrained"])
        self.downsample = KANHead(self.backbone.out_dim, [256], cfg["model"]["kan_downsample_dim"],
                                   force_mlp_fallback=force_mlp_fallback)
        self.scalar_head = KANHead(cfg["model"]["kan_downsample_dim"], [32], 1,
                                    force_mlp_fallback=force_mlp_fallback)

    def embed(self, x):
        feat = self.backbone(x)
        return self.downsample(feat)

    def forward(self, x):
        emb = self.embed(x)
        score = self.scalar_head(emb)
        return score.squeeze(-1)


class TargetRegionBranchModel(BranchModel):
    """Same backbone/heads as BranchModel, but forward() expects a batch of
    [B, num_regions, C, H, W] region crops and mean-pools region embeddings before
    scoring -- this is what lets the branch reflect target-region usability rather
    than whole-image aesthetics."""

    def embed(self, x):
        b, n, c, h, w = x.shape
        flat = x.view(b * n, c, h, w)
        feat = self.backbone(flat)
        emb = self.downsample(feat)
        emb = emb.view(b, n, -1).mean(dim=1)
        return emb
