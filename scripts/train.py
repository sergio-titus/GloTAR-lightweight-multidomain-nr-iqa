
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import Image
from scipy.stats import spearmanr, pearsonr, kendalltau
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_config, set_seed, get_device
from src.datasets import default_transform
from src.saliency import saliency_region_proposals
from src.models.global_target_iqa import GlobalTargetIQAModel


ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# DATASET
# ============================================================

class MultiDomainIQADataset(Dataset):

    def __init__(self, manifest, split, cfg):

        self.df = (
            manifest[manifest["split"] == split]
            .reset_index(drop=True)
        )

        self.global_tf = default_transform(
            tuple(
                cfg["preprocessing"]["global_branch_resize"]
            )
        )

        self.target_tf = default_transform(
            tuple(
                cfg["preprocessing"]["target_branch_crop"]
            )
        )

        self.max_regions = int(
            cfg["preprocessing"]["max_target_regions_per_image"]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        r = self.df.iloc[idx]

        img = Image.open(
            r["image_path"]
        ).convert("RGB")

        # -------------------------------
        # Global branch
        # -------------------------------

        g = self.global_tf(img)

        # -------------------------------
        # Target / salient-region branch
        # -------------------------------

        regions = saliency_region_proposals(
            np.array(img),
            max_regions=self.max_regions
        )[:self.max_regions]

        crops = []

        for box in regions:

            x1, y1, x2, y2 = map(int, box)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.width, x2)
            y2 = min(img.height, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img.crop(
                (x1, y1, x2, y2)
            )

            crops.append(
                self.target_tf(crop)
            )

        if len(crops) == 0:
            crops = [
                self.target_tf(img)
            ]

        template = crops[0]

        while len(crops) < self.max_regions:
            crops.append(
                torch.zeros_like(template)
            )

        crops = crops[:self.max_regions]

        t = torch.stack(
            crops,
            dim=0
        )

        return {
            "global_image": g,
            "target_regions": t,
            "mos": torch.tensor(
                float(r["mos"]),
                dtype=torch.float32
            ),
            "dataset": str(r["dataset"]),
        }


# ============================================================
# METRICS
# ============================================================

def metrics(pred, target):

    pred = np.asarray(
        pred,
        dtype=np.float64
    )

    target = np.asarray(
        target,
        dtype=np.float64
    )

    return {
        "srcc": float(
            spearmanr(pred, target).statistic
        ),
        "plcc": float(
            pearsonr(pred, target).statistic
        ),
        "krcc": float(
            kendalltau(pred, target).statistic
        ),
        "rmse": float(
            np.sqrt(
                np.mean(
                    (pred-target)**2
                )
            )
        ),
        "mae": float(
            np.mean(
                np.abs(
                    pred-target
                )
            )
        ),
    }


# ============================================================
# TRAIN
# ============================================================

def train_epoch(
    model,
    loader,
    optimizer,
    device
):

    model.train()

    total = 0.0
    n = 0

    for batch in tqdm(
        loader,
        desc="train",
        leave=False
    ):

        g = batch[
            "global_image"
        ].to(device)

        t = batch[
            "target_regions"
        ].to(device)

        y = batch[
            "mos"
        ].to(device)

        optimizer.zero_grad()

        pred, _ = model(
            g,
            t
        )

        loss = torch.nn.functional.mse_loss(
            pred,
            y
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0
        )

        optimizer.step()

        total += (
            loss.item()
            * y.size(0)
        )

        n += y.size(0)

    return total / max(n, 1)


# ============================================================
# VALIDATION
# ============================================================

def evaluate(
    model,
    loader,
    device
):

    model.eval()

    records = []

    with torch.no_grad():

        for batch in tqdm(
            loader,
            desc="val",
            leave=False
        ):

            g = batch[
                "global_image"
            ].to(device)

            t = batch[
                "target_regions"
            ].to(device)

            pred, _ = model(
                g,
                t
            )

            pred = (
                pred
                .detach()
                .cpu()
                .numpy()
            )

            y = (
                batch["mos"]
                .numpy()
            )

            datasets = batch[
                "dataset"
            ]

            for p, yy, ds in zip(
                pred,
                y,
                datasets
            ):

                records.append({
                    "dataset": ds,
                    "pred": float(p),
                    "target": float(yy),
                })

    df = pd.DataFrame(
        records
    )

    per_dataset = {}

    for dataset, sub in df.groupby(
        "dataset"
    ):

        per_dataset[
            dataset
        ] = metrics(
            sub["pred"],
            sub["target"]
        )

    macro_srcc = float(
        np.mean([
            m["srcc"]
            for m
            in per_dataset.values()
        ])
    )

    macro_plcc = float(
        np.mean([
            m["plcc"]
            for m
            in per_dataset.values()
        ])
    )

    return (
        per_dataset,
        macro_srcc,
        macro_plcc
    )


# ============================================================
# MAIN
# ============================================================

def main():

    cfg = load_config(
        str(
            ROOT
            / "configs/default.yaml"
        )
    )

    set_seed(
        cfg["seed"]
    )

    device = get_device(
        cfg
    )

    print(
        "Device:",
        device
    )

    manifest_path = ROOT / cfg["paths"]["manifest_path"]

    manifest = pd.read_csv(
        manifest_path
    )

    print(
        "Datasets:",
        sorted(
            manifest[
                "dataset"
            ].unique()
        )
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_ds = MultiDomainIQADataset(
        manifest,
        "train",
        cfg
    )

    val_ds = MultiDomainIQADataset(
        manifest,
        "val",
        cfg
    )

    # --------------------------------------------------------
    # DATASET-BALANCED SAMPLING
    # --------------------------------------------------------

    counts = (
        train_ds.df[
            "dataset"
        ].value_counts()
    )

    print(
        "\nTraining counts:"
    )

    print(
        counts
    )

    sample_weights = (
        train_ds.df[
            "dataset"
        ]
        .map(
            lambda x:
            1.0 / counts[x]
        )
        .astype(float)
        .values
    )

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(
            sample_weights
        ),
        num_samples=len(
            train_ds
        ),
        replacement=True
    )

    batch_size = int(
        cfg["training"]["batch_size"]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    # IMPORTANT:
    # Clean experiment.
    # ImageNet initialization only.
    # Do NOT load generic/UAV checkpoints.
    model = GlobalTargetIQAModel(
        cfg,
        use_distributional_head=False
    ).to(device)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "\nParameters:",
        f"{total_params/1e6:.3f} M"
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=1e-4
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=20,
            eta_min=5e-7
        )
    )

    # 20 epochs initially.
    # Macro-SRCC chooses checkpoint.
    epochs = cfg["training"]["epochs"]

    out_path = (
        ROOT
        / "results/checkpoints/"
        "multidomain_iqa.pt"
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    best_macro = -1.0

    print(
        "\n"
        + "="*72
    )

    print(
        "MULTI-DOMAIN GENERAL IQA TRAINING"
    )

    print(
        "="*72
    )

    # --------------------------------------------------------
    # TRAIN LOOP
    # --------------------------------------------------------

    for epoch in range(
        1,
        epochs + 1
    ):

        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device
        )

        (
            per_dataset,
            macro_srcc,
            macro_plcc
        ) = evaluate(
            model,
            val_loader,
            device
        )

        scheduler.step()

        print(
            f"\n[epoch {epoch:02d}/{epochs}] "
            f"loss={loss:.4f} "
            f"macro_SRCC={macro_srcc:.4f} "
            f"macro_PLCC={macro_plcc:.4f}"
        )

        for name in sorted(
            per_dataset
        ):

            m = per_dataset[
                name
            ]

            print(
                f"  {name:10s} "
                f"SRCC={m['srcc']:.4f} "
                f"PLCC={m['plcc']:.4f}"
            )

        # ----------------------------------------------------
        # MACRO-SRCC CHECKPOINT SELECTION
        # ----------------------------------------------------

        if macro_srcc > best_macro:

            best_macro = (
                macro_srcc
            )

            torch.save(
                {
                    "model":
                        model.state_dict(),

                    "epoch":
                        epoch,

                    "macro_srcc":
                        macro_srcc,

                    "macro_plcc":
                        macro_plcc,

                    "per_dataset":
                        per_dataset,

                    "protocol":
                        "five_dataset_balanced_ImageNet_init",
                },
                out_path
            )

            print(
                "  ✓ SAVED BEST"
            )

    print(
        "\n"
        + "="*72
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "="*72
    )

    print(
        "Best macro SRCC:",
        f"{best_macro:.4f}"
    )

    print(
        "Checkpoint:",
        out_path
    )


if __name__ == "__main__":
    main()
