# Colab launch

Minimal GPU path for the Cloude transformer JEPA.

## 1. Start Colab

- Runtime -> Change runtime type -> GPU
- Confirm with:

```bash
!nvidia-smi
```

## 2. Clone the repo

```bash
%cd /content
!git clone <YOUR_GITHUB_REPO_URL> Phy-L-Jepa
%cd /content/Phy-L-Jepa
```

## 3. Install runtime deps

```bash
!pip -q install numpy
```

## 4. Attach the data

If the ColoPola NPZ files are in Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then point the training script to the mounted folder.

## 5. Train

```bash
!python train_jepa_cpu_150.py --data-root /content/drive/MyDrive/GIGADATASET_COLAB_NPZ --epochs 150 --batch-size 256 --output-dir /content/Phy-L-Jepa/results/phys_jepa_cloude_transformer
```

## 6. Probe the frozen JEPA

```bash
!python train_probe_mlp.py --data-root /content/drive/MyDrive/GIGADATASET_COLAB_NPZ --jepa-ckpt /content/phy-l-jepa/results/phys_jepa_cloude_transformer/phys_jepa_cloude_transformer/latest.pth.tar --output-dir /content/Phy-L-Jepa/results/phys_jepa_probe_suite_full
```
