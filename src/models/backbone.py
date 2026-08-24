"""Lightweight backbone wrapper. Supports loading ImageNet weights or a custom
IQA-pretrained checkpoint (see scripts/03_train_branch.py --pretrain_backbone)."""
import torch
import torch.nn as nn
import torchvision.models as tvm


class MobileNetV3Backbone(nn.Module):
    def __init__(self, variant: str = "mobilenet_v3_small", pretrained: bool = True):
        super().__init__()
        if variant == "mobilenet_v3_small":
            net = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
            self.out_dim = 576
        elif variant == "mobilenet_v3_large":
            net = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
            self.out_dim = 960
        else:
            raise ValueError(f"unknown backbone variant {variant}")

        self.features = net.features
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        feats = self.features(x)
        pooled = self.pool(feats).flatten(1)
        return pooled  # [B, out_dim]

    def forward_with_feature_map(self, x):
        """Returns both the pooled embedding and the last conv feature map,
        needed for Grad-CAM style saliency in Step 6."""
        feats = self.features(x)
        pooled = self.pool(feats).flatten(1)
        return pooled, feats

    def load_iqa_pretrained(self, checkpoint_path: str):
        state = torch.load(checkpoint_path, map_location="cpu")
        missing, unexpected = self.load_state_dict(state, strict=False)
        print(f"[MobileNetV3Backbone] loaded IQA-pretrained weights from {checkpoint_path} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
