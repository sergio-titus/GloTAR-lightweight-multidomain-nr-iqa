# Lightweight Multi-Domain No-Reference Image Quality Assessment

Official implementation of a lightweight multi-domain no-reference image quality assessment framework based on global image information and salient-region feature fusion.

## Architecture

The framework contains two complementary branches:

1. **Global branch** — models overall image quality.
2. **Salient-region branch** — models perceptually important local regions.

The two representations are fused using a lightweight regression head to predict image Mean Opinion Score (MOS).

The normalized quality score is:

$$
Q = \frac{\widehat{MOS} - MOS_{\min}}{MOS_{\max} - MOS_{\min}}
$$

where $MOS_{\min}=1$ and $MOS_{\max}=5$. Therefore:

$$
Q = \frac{\widehat{MOS}-1}{4}.
$$

## Datasets

The model was trained and evaluated across five IQA datasets:

- BIQ2021
- KADID-10K
- KonIQ-10K
- TID2013
- **Drone-IQA GC 2026: Target-Aware Image Quality Assessment for Low-Altitude UAV Images**

## Reproducibility

The repository provides the code, experimental configurations, fixed
data-partition manifests, and evaluation outputs used to support the
reported experiments.

Available reproducibility material includes:

- fixed train/validation/test assignments for all datasets;
- reference-content-disjoint partitions for TID2013 and KADID-10K;
- the exact internal Drone-IQA partition used in the experiments;
- training and evaluation configurations;
- seed-level results for repeated GloTAR training runs;
- the experimental protocol and implementation details;
- primary held-out evaluation results.

The original image datasets are not redistributed. Users should obtain
them from their respective official sources and use the provided split
manifests to reproduce the experimental partitions.

### Drone-IQA protocol used in this repository

Only the publicly released **Drone-IQA GC 2026 training images and `train.csv` annotations** were used.

The official validation and held-out test annotations are not publicly available, so this repository does **not** report results on the official hidden Drone-IQA challenge test set.

The 3,600 publicly labeled Drone-IQA training images were internally partitioned into:

- 2,520 images for training
- 540 images for validation
- 540 images for internal testing

For the final five-dataset multi-domain experiment, the former calibration subset was merged back into training, resulting in:

- **2,520 training images**
- **540 validation images**
- **540 internal test images**

The reported Drone-IQA SRCC/PLCC values therefore refer to this **internal split of the public training set**, not to the official hidden challenge test set.

## Final Held-Out Results

| Dataset | SRCC | PLCC | KRCC | RMSE | MAE |
|---|---:|---:|---:|---:|---:|
| BIQ2021 | 0.7210 | 0.7617 | 0.5322 | 0.1186 | 0.0927 |
| Drone-IQA GC 2026 (internal test split) | 0.9016 | 0.9002 | 0.7242 | 0.0874 | 0.0685 |
| KADID-10K | 0.8402 | 0.8327 | 0.6444 | 0.1515 | 0.1203 |
| KonIQ-10K | 0.7880 | 0.8238 | 0.5966 | 0.0945 | 0.0722 |
| TID2013 | 0.7743 | 0.8172 | 0.5836 | 0.1232 | 0.1018 |
| **Macro Average** | **0.8050** | **0.8271** | **0.6162** | **0.1151** | **0.0911** |

## Model Complexity

- Parameters: **6.525 M**
- Checkpoint size: **25.31 MB**
- Global input: **384 × 384**
- Salient-region crops: **224 × 224**
- Maximum salient regions: **6**

## Runtime

### Tesla P100 GPU

- Model-only latency: **14.80 ms**
- Model-only throughput: **67.55 FPS**
- End-to-end latency: **70.91 ms**
- End-to-end throughput: **14.10 FPS**

### CPU

- Model-only latency: **52.36 ms**
- Model-only throughput: **19.10 FPS**
- End-to-end latency: **113.70 ms**
- End-to-end throughput: **8.80 FPS**

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/lightweight-multidomain-nr-iqa.git
cd lightweight-multidomain-nr-iqa
pip install -r requirements.txt
```

## Single-Image Inference

```bash
python scripts/infer.py path/to/image.jpg
```

Example output:

```text
Predicted MOS: 3.8105
Normalized Q : 0.7026
```

## Training

```bash
python scripts/train.py
```

Dataset paths and training parameters can be configured in `configs/default.yaml`.

## Evaluation

```bash
python scripts/evaluate.py
```

## Citation

A citation entry will be added after publication.

## License

License information will be provided with the official release.
