from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from colopola_dataset import ColoPolaDataset
from physics_features import CloudeTransformerEncoder
from train_probe_mlp import ProbeHead, build_base_dataset, encode_dataset, standardize


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(r"C:\Users\ayoub\Desktop\Stage\Project\data\GIGADATASET_COLAB_NPZ")
DEFAULT_JEPA_CKPT = PROJECT_ROOT / "results" / "phys_jepa_cloude_transformer" / "phys_jepa_cloude_transformer" / "latest.pth.tar"
DEFAULT_SUITE_DIR = PROJECT_ROOT / "results" / "phys_jepa_probe_suite_full"


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


@torch.no_grad()
def compute_metrics(head: nn.Module, features: torch.Tensor, labels: torch.Tensor) -> dict:
    head.eval()
    loader = DataLoader(TensorDataset(features, labels), batch_size=1024, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0
    tp = fp = fn = 0
    for x, y in loader:
        logits = head(x)
        loss = criterion(logits, y)
        preds = logits.argmax(dim=1)
        bs = int(y.shape[0])
        total_loss += float(loss.item()) * bs
        total += bs
        correct += int((preds == y).sum().item())
        tp += int(((preds == 1) & (y == 1)).sum().item())
        fp += int(((preds == 1) & (y == 0)).sum().item())
        fn += int(((preds == 0) & (y == 1)).sum().item())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_samples": total,
    }


def load_encoder(ckpt_path: Path, image_size: int, device: torch.device) -> nn.Module:
    encoder = CloudeTransformerEncoder(
        patch_size=1,
        embed_dim=64,
        mlp_hidden_dim=128,
        num_heads=4,
        depth=2,
        dropout=0.1,
        image_size=image_size,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model_state = state.get("model", state)
    encoder_state = {
        k.removeprefix("context_encoder."): v
        for k, v in model_state.items()
        if k.startswith("context_encoder.")
    }
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Unexpected checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder


def evaluate_kind(kind: str, suite_dir: Path, test_features: torch.Tensor, test_labels: torch.Tensor) -> dict:
    probe_path = suite_dir / kind / "best.pth.tar"
    payload = torch.load(probe_path, map_location="cpu")
    head = ProbeHead(test_features.shape[1], kind=kind, hidden_dim=32, dropout=0.2)
    head.load_state_dict(payload["head"])
    metrics = compute_metrics(head, test_features, test_labels)
    out = {"kind": kind, "checkpoint": str(probe_path), "metrics": metrics}
    save_json(suite_dir / kind / "test_only_metrics.json", out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--jepa-ckpt", type=Path, default=DEFAULT_JEPA_CKPT)
    parser.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE_DIR)
    parser.add_argument("--kind", choices=("linear", "mlp", "both"), default="both")
    args = parser.parse_args()

    device = torch.device("cpu")
    test_ds = build_base_dataset(args.data_root, "test", None, smoke_test=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0, drop_last=False)

    sample, _ = test_ds[0]
    image_size = int(max(sample.shape[1], sample.shape[2]))
    encoder = load_encoder(args.jepa_ckpt, image_size=image_size, device=device)

    standardizer = torch.load(args.suite_dir / "standardizer.pt", map_location="cpu")
    mean = standardizer["mean"]
    std = standardizer["std"]

    test_features, test_labels = encode_dataset(encoder, test_loader, device)
    test_features = standardize(test_features, mean, std)

    kinds = ["linear", "mlp"] if args.kind == "both" else [args.kind]
    results = [evaluate_kind(kind, args.suite_dir, test_features, test_labels) for kind in kinds]
    save_json(args.suite_dir / "test_only_summary.json", {"jepa_ckpt": str(args.jepa_ckpt), "results": results})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
