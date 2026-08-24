
import torch
import torch.nn as nn

from .branches import BranchModel, TargetRegionBranchModel
from .kan import KANHead


class GlobalTargetIQAModel(nn.Module):
    """
    Lightweight two-branch NR-IQA model.

    Branches:
      1. Global-image quality branch
      2. Salient/target-region quality branch

    The two embeddings are concatenated and mapped to a scalar MOS prediction.
    """

    def __init__(
        self,
        cfg,
        force_mlp_fallback=False,
        use_distributional_head=False,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Two independent MobileNetV3-based IQA branches
        # ----------------------------------------------------

        self.global_branch = BranchModel(
            cfg,
            force_mlp_fallback
        )

        self.target_branch = TargetRegionBranchModel(
            cfg,
            force_mlp_fallback
        )

        d = int(
            cfg["model"]["kan_downsample_dim"]
        )

        self.use_distributional_head = (
            use_distributional_head
        )

        # This general-model experiment uses the scalar head.
        if use_distributional_head:
            raise NotImplementedError(
                "The clean multi-domain experiment uses "
                "use_distributional_head=False."
            )

        # ----------------------------------------------------
        # Fusion head
        # ----------------------------------------------------

        self.fusion_head = KANHead(
            2 * d,
            [256, 128],
            1,
            force_mlp_fallback=force_mlp_fallback
        )

    def freeze_branches(self, freeze=True):

        for branch in [
            self.global_branch,
            self.target_branch,
        ]:
            for p in branch.parameters():
                p.requires_grad = not freeze

    def forward(
        self,
        global_img,
        target_regions,
    ):

        # Global embedding
        g = self.global_branch.embed(
            global_img
        )

        # Salient/target-region embedding
        t = self.target_branch.embed(
            target_regions
        )

        # Fusion
        fused = torch.cat(
            [g, t],
            dim=1
        )

        score = self.fusion_head(
            fused
        ).squeeze(-1)

        return score, {
            "global_emb": g,
            "target_emb": t,
        }
