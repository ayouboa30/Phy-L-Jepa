# Colab launch

Minimal GPU path for the masked image JEPA.

## Start Colab

- Runtime -> Change runtime type -> GPU
- Confirm with:

```bash
!nvidia-smi
```

## Clone or refresh the repo

```bash
%cd /content
!test -d Phy-L-Jepa || git clone <YOUR_GITHUB_REPO_URL> Phy-L-Jepa
%cd /content/Phy-L-Jepa
!git pull
```

## Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Train masked image JEPA

```bash
!python train_jepa_image_gpu.py --data-root /content/drive/MyDrive/GIGADATASET_COLAB_NPZ --epochs 150 --batch-size 512 --embed-dim 128 --depth 4 --num-heads 8 --mlp-hidden-dim 256 --mask-ratio 0.6 --output-dir /content/Phy-L-Jepa/results/masked_image_jepa_gpu
```

## Train probes

```bash
!python train_probe_mlp.py --encoder-type image --data-root /content/drive/MyDrive/GIGADATASET_COLAB_NPZ --jepa-ckpt /content/Phy-L-Jepa/results/masked_image_jepa_gpu/masked_image_jepa/latest.pth.tar --output-dir /content/Phy-L-Jepa/results/masked_image_probe_gpu --epochs 40 --batch-size 512 --hidden-dim 64 --dropout 0.25 --patience 8
```

## Test only

```bash
!python eval_probe_mlp.py --encoder-type image --data-root /content/drive/MyDrive/GIGADATASET_COLAB_NPZ --jepa-ckpt /content/Phy-L-Jepa/results/masked_image_jepa_gpu/masked_image_jepa/latest.pth.tar --suite-dir /content/Phy-L-Jepa/results/masked_image_probe_gpu --kind both
```
