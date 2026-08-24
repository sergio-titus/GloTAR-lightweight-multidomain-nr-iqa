
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.global_target_iqa import GlobalTargetIQAModel
from src.saliency import saliency_region_proposals
from src.datasets import default_transform


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image",
        help="Path to input image"
    )

    args = parser.parse_args()

    with open(
        ROOT / "configs/default.yaml"
    ) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = GlobalTargetIQAModel(
        cfg,
        use_distributional_head=False
    ).to(device)

    checkpoint = torch.load(
        ROOT / "weights/multidomain_iqa.pt",
        map_location=device,
        weights_only=False
    )

    state = (
        checkpoint["model"]
        if isinstance(checkpoint, dict)
        and "model" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state)
    model.eval()

    image = Image.open(
        args.image
    ).convert("RGB")

    global_tf = default_transform(
        tuple(
            cfg["preprocessing"]
            ["global_branch_resize"]
        )
    )

    target_tf = default_transform(
        tuple(
            cfg["preprocessing"]
            ["target_branch_crop"]
        )
    )

    max_regions = int(
        cfg["preprocessing"]
        ["max_target_regions_per_image"]
    )

    g = (
        global_tf(image)
        .unsqueeze(0)
        .to(device)
    )

    boxes = saliency_region_proposals(
        np.array(image),
        max_regions=max_regions
    )[:max_regions]

    crops = []

    for x1, y1, x2, y2 in boxes:

        crop = image.crop(
            (
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            )
        )

        crops.append(
            target_tf(crop)
        )

    if not crops:
        crops = [
            target_tf(image)
        ]

    template = crops[0]

    while len(crops) < max_regions:
        crops.append(
            torch.zeros_like(template)
        )

    t = (
        torch.stack(
            crops[:max_regions]
        )
        .unsqueeze(0)
        .to(device)
    )

    with torch.inference_mode():

        pred, _ = model(
            g,
            t
        )

    mos = float(
        pred.squeeze().cpu()
    )

    q = float(
        np.clip(
            (mos - 1.0) / 4.0,
            0.0,
            1.0
        )
    )

    print(
        f"Predicted MOS: {mos:.4f}"
    )

    print(
        f"Normalized Q : {q:.4f}"
    )


if __name__ == "__main__":
    main()
