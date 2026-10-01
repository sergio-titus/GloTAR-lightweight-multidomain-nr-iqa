# Experimental Protocol

## Data partitions

All reported experiments use fixed train/validation/test partitions.
The exact image-level assignments are provided under `data/splits/`.

Drone-IQA contains 3,600 publicly labelled images in the protocol used
in this work:

- Training: 2,520
- Validation: 540
- Internal test: 540

The same fixed partitions were used for all repeated runs.

For TID2013 and KADID-10K, reference-content-disjoint partitions were
used.

## Training

- Optimizer: AdamW
- Learning rate: 5e-5
- Weight decay: 1e-4
- Batch size: 32
- Maximum epochs: 20
- Gradient clipping: maximum norm 5
- Scheduler: CosineAnnealingLR
- T_max: 20
- eta_min: 5e-7
- Loss: MSE
- Checkpoint selection: validation macro-SRCC

## Dataset-balanced sampling

For a training dataset d containing n_d samples, each image receives
sampling weight:

w_i = 1 / n_d

Weighted sampling is performed with replacement.

Each epoch contains 27,983 sampled instances, equal to the total
number of training instances.

## Quality-score normalization

Q = (MOS - 1) / 4

Higher Q corresponds to higher perceptual quality.

SRCC, PLCC, RMSE and MAE are computed directly without nonlinear or
logistic post-fitting.

## KAN configuration

The KAN implementation uses `efficient-kan`.

The experiments used the library defaults:

- grid size: 5
- spline order: 3
- scale_noise: 0.1
- scale_base: 1.0
- scale_spline: 1.0
- base activation: SiLU
- grid_eps: 0.02
- grid range: [-1, 1]
