# Phy-L JEPA

Mueller-matrix JEPA pipeline for the ColoPola dataset.

## Scope

This repository is organized around a compact, publication-style training stack:

- direct physical features from the coherency matrix real and imaginary parts
- lightweight transformer context encoder for the Cloude feature path
- CPU training entrypoint plus Colab/GPU-friendly launch path
- ColoPola NPZ training data
- masked JEPA and physics-retention JEPA variants

The goal is to keep the codebase focused on the stage work and on a reproducible training workflow.

## Main components

- `colopola_dataset.py`: ColoPola NPZ loader
- `physics_features.py`: coherency extractor plus MLP and transformer encoders
- `architectures.py`: lightweight Mueller encoder and MLP predictor
- `hybrid_physics_jepa.py`: masked JEPA with direct physical retention
- `pretrain_adaptive_hybrid.py`: CPU-only training entrypoint
- `train_jepa_cpu_150.py`: dedicated 150-epoch training launcher for the Cloude transformer JEPA
- `train_probe_mlp.py`: frozen-JEPA linear and MLP probe suite for healthy/cancer prediction

## Data

The default dataset path used by the new launcher is:

`C:\Users\ayoub\Desktop\Stage\Project\data\GIGADATASET_COLAB_NPZ`

Expected files include the train and test NPZ splits already present in that directory.

## Training

Run the 150-epoch Cloude transformer JEPA:

```bash
py -3.13 train_jepa_cpu_150.py --epochs 150 --output-dir results/phys_jepa_cloude_transformer
```

Outputs are written under `results/phys_jepa_cloude_transformer/`.

## Alternative entrypoint

The more general CPU-only pretraining script is:

```bash
py -3.13 pretrain_adaptive_hybrid.py --config config.yaml --epochs 150
```

It uses the same ColoPola loader and direct coherency features.

## Colab

See `COLAB.md` for a minimal GPU launch path in Google Colab.

## Repository layout

- `config.yaml`: default CPU configuration
- `config.mueller.local.yaml`: local override
- `results/`: training logs and checkpoints, ignored by Git
- `benchmark_results/`: legacy benchmark outputs kept for reference, ignored by Git

## Notes

- The transformer path is intentionally lightweight for CPU or Colab execution.
- The training pipeline assumes Mueller tensors with 16 channels.
- No image assets are required for the main workflow.