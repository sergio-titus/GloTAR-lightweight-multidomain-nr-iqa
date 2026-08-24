
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from PIL import Image
from scipy.stats import spearmanr, pearsonr, kendalltau
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.global_target_iqa import GlobalTargetIQAModel
from src.saliency import saliency_region_proposals
from src.datasets import default_transform


def compute_metrics(y, p):

    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)

    return {
        "srcc": spearmanr(y, p).statistic,
        "plcc": pearsonr(y, p).statistic,
        "krcc": kendalltau(y, p).statistic,
        "rmse": np.sqrt(np.mean((y - p) ** 2)),
        "mae": np.mean(np.abs(y - p)),
    }


def prepare_image(
    image,
    global_transform,
    target_transform,
    max_regions,
    device
):

    g = global_transform(
        image
    ).unsqueeze(0).to(device)

    boxes = saliency_region_proposals(
        np.array(image),
        max_regions=max_regions
    )[:max_regions]

    crops = []

    for x1, y1, x2, y2 in boxes:

        x1, y1, x2, y2 = map(
            int,
            [x1, y1, x2, y2]
        )

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(image.width, x2)
        y2 = min(image.height, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = image.crop(
            (x1, y1, x2, y2)
        )

        crops.append(
            target_transform(crop)
        )

    if not crops:
        crops = [
            target_transform(image)
        ]

    template = crops[0]

    while len(crops) < max_regions:
        crops.append(
            torch.zeros_like(template)
        )

    t = torch.stack(
        crops[:max_regions]
    ).unsqueeze(0).to(device)

    return g, t


def main():

    with open(
        ROOT / "configs/default.yaml"
    ) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        ROOT / "weights/multidomain_iqa.pt",
        map_location=device,
        weights_only=False
    )

    model = GlobalTargetIQAModel(
        cfg,
        use_distributional_head=False
    ).to(device)

    state = (
        checkpoint["model"]
        if isinstance(checkpoint, dict)
        and "model" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state)
    model.eval()

    manifest_path = (
        ROOT / "data/multidomain_iqa_manifest.csv"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            "Place multidomain_iqa_manifest.csv "
            "inside data/ before evaluation."
        )

    manifest = pd.read_csv(
        manifest_path
    )

    test = manifest[
        manifest["split"] == "test"
    ].copy()

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

    records = []

    with torch.inference_mode():

        for _, row in tqdm(
            test.iterrows(),
            total=len(test)
        ):

            image = Image.open(
                row["image_path"]
            ).convert("RGB")

            g, t = prepare_image(
                image,
                global_tf,
                target_tf,
                max_regions,
                device
            )

            pred, _ = model(
                g,
                t
            )

            mos = float(
                pred.squeeze().cpu()
            )

            q = np.clip(
                (mos - 1.0) / 4.0,
                0.0,
                1.0
            )

            records.append({
                "dataset":
                    row["dataset"],

                "true_quality":
                    row["quality_score"],

                "predicted_quality":
                    q,
            })

    predictions = pd.DataFrame(
        records
    )

    rows = []

    for dataset, sub in predictions.groupby(
        "dataset"
    ):

        m = compute_metrics(
            sub["true_quality"],
            sub["predicted_quality"]
        )

        rows.append({
            "dataset": dataset,
            "n": len(sub),
            **m
        })

    metrics = pd.DataFrame(
        rows
    )

    macro = {
        "dataset": "macro_average",
        "n": metrics["n"].sum()
    }

    for column in [
        "srcc",
        "plcc",
        "krcc",
        "rmse",
        "mae"
    ]:
        macro[column] = (
            metrics[column].mean()
        )

    metrics = pd.concat(
        [
            metrics,
            pd.DataFrame([macro])
        ],
        ignore_index=True
    )

    print(metrics.round(4))

    metrics.to_csv(
        ROOT / "results/evaluation.csv",
        index=False
    )


if __name__ == "__main__":
    main()
