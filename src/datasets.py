"""
Dataset classes for the three branches. All datasets return a dict:
    {"image": Tensor[C,H,W], "mos": float or None, "meta": dict}

so a single training loop in scripts/03_train_branch.py can consume any of them.
"""
from __future__ import annotations

import os
import glob
import json
import csv
from pathlib import Path
from typing import Optional, Callable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def default_transform(size):
    return T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class VisDroneDataset(Dataset):
    """Raw VisDrone2019-DET imagery. No MOS labels -- used as (a) a pristine image
    source for synthetic distortion generation, and (b) optional unlabeled/weakly
    supervised pretraining for the global branch."""

    def __init__(self, root: str, split: str = "train", transform: Optional[Callable] = None):
        split_dir_map = {
            "train": "VisDrone2019-DET-train",
            "val": "VisDrone2019-DET-val",
            "test-dev": "VisDrone2019-DET-test-dev",
            "test-challenge": "VisDrone2019-DET-test-challenge",
        }
        assert split in split_dir_map, f"unknown split {split}"
        img_dir = Path(root) / split_dir_map[split] / "images"
        self.paths = sorted(glob.glob(str(img_dir / "*.jpg")))
        if len(self.paths) == 0:
            # helpful message rather than a silent empty dataset
            print(f"[VisDroneDataset] WARNING: no images found under {img_dir}. "
                  f"Check configs/config.yaml -> paths.visdrone_root")
        self.transform = transform or default_transform((384, 384))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        return {"image": tensor, "mos": None, "meta": {"path": path, "source": "visdrone"}}


class DroneIQADataset(Dataset):
    """ICME 2026 Drone-IQA GC benchmark. Expects an annotation file with per-image
    global/target/background MOS. TODO: confirm the exact annotation filename/format
    once train.tar is unpacked and adjust `_load_labels` accordingly -- competitions
    commonly ship either a single labels.csv/json or per-image sidecar files."""

    def __init__(self, root: str, split: str = "train", label_target: str = "global",
                 transform: Optional[Callable] = None):
        assert label_target in ("global", "target", "background")
        self.root = Path(root)
        self.label_target = label_target
        self.transform = transform or default_transform((384, 384))
        self.records = self._load_labels(split)

    # Confirmed Drone-IQA GC CSV schema (train split):
    # filename, global_quality_mean, global_quality_std,
    #           target_quality_mean, target_quality_std,
    #           background_quality_mean, background_quality_std
    # NOTE: the official "val" split only ships this CSV, no images -- it can't be used
    # for local validation. Use scripts/01_prepare_dataset.py to carve an internal
    # train/val split out of the labeled "train" data instead (see split_internal_val()).
    MEAN_COL = {"global": "global_quality_mean", "target": "target_quality_mean",
                "background": "background_quality_mean"}
    STD_COL = {"global": "global_quality_std", "target": "target_quality_std",
               "background": "background_quality_std"}

    def _load_labels(self, split: str):
        # split here means "train" (the only split with images) or an internal
        # sub-split name produced by split_internal_val(), e.g. "train_internal_train.csv"
        candidates = [
            self.root / f"{split}.csv",
            self.root / f"{split}_labels.csv",
            self.root / "train.csv",
            self.root / "labels.csv",
        ]
        for c in candidates:
            if c.exists():
                return self._read_csv(c)
        found = sorted(str(p) for p in self.root.glob("*.csv"))
        print(f"[DroneIQADataset] WARNING: no matching label file for split='{split}' in {self.root}. "
              f"CSV files found there: {found}. Falling back to an empty dataset.")
        return []

    def _read_csv(self, path: Path):
        mean_col, std_col = self.MEAN_COL[self.label_target], self.STD_COL[self.label_target]
        records = []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if not row.get(mean_col):
                    continue
                img_path = self.root / "images" / row["filename"]
                if not img_path.exists():
                    # official val CSV has no matching images -- skip rather than crash
                    continue
                std_val = row.get(std_col)
                records.append({
                    "image_path": str(img_path),
                    "mos": float(row[mean_col]),
                    "mos_std": float(std_val) if std_val not in (None, "") else None,
                })
        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec["image_path"]).convert("RGB")
        tensor = self.transform(img)
        return {"image": tensor, "mos": rec["mos"],
                "meta": {"path": rec["image_path"], "source": "drone_iqa", "target": self.label_target}}


class SyntheticDistortionDataset(Dataset):
    """Consumes the manifest produced by scripts/02_generate_synthetic_distortions.py.
    Each row: distorted_image_path, distortion_type, severity, pseudo_mos."""

    def __init__(self, manifest_csv: str, transform: Optional[Callable] = None):
        self.records = []
        if os.path.exists(manifest_csv):
            with open(manifest_csv, newline="") as f:
                self.records = list(csv.DictReader(f))
        else:
            print(f"[SyntheticDistortionDataset] WARNING: manifest not found at {manifest_csv}. "
                  f"Run scripts/02_generate_synthetic_distortions.py first.")
        self.transform = transform or default_transform((224, 224))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec["distorted_image_path"]).convert("RGB")
        tensor = self.transform(img)
        return {
            "image": tensor,
            "mos": float(rec["pseudo_mos"]),
            "meta": {"distortion_type": rec["distortion_type"], "severity": float(rec["severity"]),
                     "source": "synthetic"},
        }


class FusionDataset(Dataset):
    """Step 5: for each Drone-IQA GC image, produces all three branch inputs at once
    (global resize, distortion-branch crop, target-region crops) plus the global MOS
    as the fusion target. Used by scripts/04_train_fusion.py."""

    def __init__(self, drone_iqa_root: str, split: str, region_proposer, cfg: dict):
        self.global_ds = DroneIQADataset(
            drone_iqa_root, split=split, label_target="global",
            transform=default_transform(tuple(cfg["preprocessing"]["global_branch_resize"])),
        )
        self.distortion_transform = default_transform(tuple(cfg["preprocessing"]["distortion_branch_crop"]))
        self.target_transform = default_transform(tuple(cfg["preprocessing"]["target_branch_crop"]))
        self.region_proposer = region_proposer
        self.max_regions = cfg["preprocessing"]["max_target_regions_per_image"]

    def __len__(self):
        return len(self.global_ds)

    def __getitem__(self, idx):
        rec = self.global_ds.records[idx]
        img = Image.open(rec["image_path"]).convert("RGB")

        global_tensor = self.global_ds.transform(img)
        distortion_tensor = self.distortion_transform(img)

        regions = self.region_proposer(np.array(img))[: self.max_regions]
        crops = [self.target_transform(img.crop(box)) for box in regions]
        if len(crops) == 0:
            crops = [self.target_transform(img)]
        target_tensor = torch.stack(crops, dim=0)

        return {
            "global_image": global_tensor,
            "distortion_image": distortion_tensor,
            "target_regions": target_tensor,
            "mos": rec["mos"],
            "mos_std": rec.get("mos_std"),
            "meta": {"path": rec["image_path"]},
        }


def fusion_collate_fn(batch):
    """Custom collate needed because target_regions can have a different number of
    crops per image -- pads to the batch max rather than requiring a fixed count."""
    max_regions = max(item["target_regions"].shape[0] for item in batch)
    padded = []
    for item in batch:
        t = item["target_regions"]
        if t.shape[0] < max_regions:
            pad = t[-1:].repeat(max_regions - t.shape[0], 1, 1, 1)
            t = torch.cat([t, pad], dim=0)
        padded.append(t)
    return {
        "global_image": torch.stack([b["global_image"] for b in batch]),
        "distortion_image": torch.stack([b["distortion_image"] for b in batch]),
        "target_regions": torch.stack(padded),
        "mos": [b["mos"] for b in batch],
        "mos_std": [b["mos_std"] for b in batch],
    }


class TargetRegionDataset(Dataset):
    """Crops candidate target regions (from src/saliency.py proposals or VisDrone GT boxes)
    and pairs them with target/background MOS from DroneIQADataset(label_target='target')."""

    def __init__(self, drone_iqa_dataset: DroneIQADataset, region_proposer, max_regions: int = 6,
                 transform: Optional[Callable] = None):
        self.base = drone_iqa_dataset
        self.region_proposer = region_proposer
        self.max_regions = max_regions
        self.transform = transform or default_transform((224, 224))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        rec = self.base.records[idx]
        img = Image.open(rec["image_path"]).convert("RGB")
        regions = self.region_proposer(np.array(img))[: self.max_regions]
        crops = []
        for (x1, y1, x2, y2) in regions:
            crop = img.crop((x1, y1, x2, y2))
            crops.append(self.transform(crop))
        if len(crops) == 0:
            crops = [self.transform(img)]
        stacked = torch.stack(crops, dim=0)  # [num_regions, C, H, W]
        return {"image": stacked, "mos": rec["mos"],
                "meta": {"path": rec["image_path"], "source": "drone_iqa_target", "num_regions": len(crops)}}
