from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from src.evaluation.metrics import evaluate_model
from src.models.model_utils import build_model
from src.training.utils import set_seed
from src.utils.config import load_config, prepare_experiment_dirs
from src.utils.logger import setup_logging, get_logger
from src.data.transforms import get_transforms


class SubsetWithTransform(Dataset):
    def __init__(self, base_dataset: ImageFolder, indices: Sequence[int], transform=None) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        path, target = self.base_dataset.samples[real_idx]
        img = self.base_dataset.loader(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def _select_subset_per_class(
    base_dataset: ImageFolder,
    max_images_per_class: Optional[int],
    seed: int = 42,
) -> Sequence[int]:
    if max_images_per_class is None:
        return list(range(len(base_dataset.samples)))

    by_class: dict[int, list[int]] = {}
    for idx, (_, target) in enumerate(base_dataset.samples):
        by_class.setdefault(target, []).append(idx)

    g = torch.Generator().manual_seed(seed)
    selected: list[int] = []
    for _, idxs in by_class.items():
        idxs_tensor = torch.tensor(idxs)
        if len(idxs) > max_images_per_class:
            perm = torch.randperm(len(idxs_tensor), generator=g)
            chosen = idxs_tensor[perm[:max_images_per_class]].tolist()
        else:
            chosen = idxs
        selected.extend(chosen)

    return selected


def _build_split(
    split_dir: Path,
    img_size: int,
    max_images_per_class: Optional[int],
    seed: int,
    is_train: bool,
) -> Dataset:
    base_ds = ImageFolder(root=str(split_dir))
    indices = _select_subset_per_class(
        base_dataset=base_ds,
        max_images_per_class=max_images_per_class,
        seed=seed,
    )
    train_tfms, eval_tfms = get_transforms(img_size=img_size)
    transform = train_tfms if is_train else eval_tfms
    return SubsetWithTransform(base_ds, indices, transform=transform)


def create_food_datasets(
    data_root: str | Path,
    img_size: int = 299,
    max_images_per_class_train: Optional[int] = None,
    max_images_per_class_val: Optional[int] = None,
    max_images_per_class_test: Optional[int] = None,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    data_root = Path(data_root)
    train_dir = data_root / "training"
    val_dir = data_root / "validation"
    test_dir = data_root / "testing"

    if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Expected 'training', 'validation' and 'testing' directories under {data_root}"
        )

    train_ds = _build_split(
        train_dir,
        img_size=img_size,
        max_images_per_class=max_images_per_class_train,
        seed=seed,
        is_train=True,
    )
    val_ds = _build_split(
        val_dir,
        img_size=img_size,
        max_images_per_class=max_images_per_class_val,
        seed=seed + 1,
        is_train=False,
    )
    test_ds = _build_split(
        test_dir,
        img_size=img_size,
        max_images_per_class=max_images_per_class_test,
        seed=seed + 2,
        is_train=False,
    )
    return train_ds, val_ds, test_ds


def create_food_dataloaders(
    data_root: str | Path,
    img_size: int = 299,
    batch_size: int = 64,
    num_workers: int = 4,
    max_images_per_class_train: Optional[int] = None,
    max_images_per_class_val: Optional[int] = None,
    max_images_per_class_test: Optional[int] = None,
    seed: int = 42,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds, val_ds, test_ds = create_food_datasets(
        data_root=data_root,
        img_size=img_size,
        max_images_per_class_train=max_images_per_class_train,
        max_images_per_class_val=max_images_per_class_val,
        max_images_per_class_test=max_images_per_class_test,
        seed=seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained checkpoint on a food classification dataset using "
            "metrics defined in src.evaluation.metrics."
        )
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/first_results_idc_inception_v3.json",
        help="Path to the JSON config file used for training",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the .pt checkpoint file (containing model_state)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate on",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config_path)
    cfg = prepare_experiment_dirs(cfg)

    # Set up logging
    log_cfg = cfg.get("logging", {})
    setup_logging(
        log_dir=log_cfg.get("log_dir", Path(cfg["output_dir"]) / "logs"),
        log_level=log_cfg.get("log_level", "INFO"),
        log_to_file=log_cfg.get("log_to_file", True),
    )
    logger = get_logger(__name__)

    # Fix random seed for reproducibility
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    logger.info(f"Seed set to {seed}")

    # Select device
    device_str = cfg.get("training", {}).get(
        "device",
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    # Load data
    data_cfg = cfg["data"]
    root_dir = data_cfg["root_dir"]
    img_size = int(data_cfg.get("img_size", 299))
    batch_size = int(data_cfg.get("batch_size", 64))
    num_workers = int(data_cfg.get("num_workers", 4))
    pin_memory = bool(data_cfg.get("pin_memory", True))

    max_train = data_cfg.get("max_images_per_class_train")
    max_val = data_cfg.get("max_images_per_class_val")
    max_test = data_cfg.get("max_images_per_class_test")

    train_loader, val_loader, test_loader = create_food_dataloaders(
        data_root=root_dir,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        max_images_per_class_train=max_train,
        max_images_per_class_val=max_val,
        max_images_per_class_test=max_test,
        seed=seed,
        pin_memory=pin_memory,
    )

    if args.split == "train":
        eval_loader = train_loader
    elif args.split == "val":
        eval_loader = val_loader
    else:
        eval_loader = test_loader

    # Build the model
    model_cfg = cfg["model"]
    model = build_model(
        name=model_cfg["arch"],
        num_classes=int(model_cfg["num_classes"]),
        pretrained=bool(model_cfg.get("pretrained", True)),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
    )
    model.to(device)

    # Load checkpoint
    ckpt_path = Path(args.checkpoint_path)
    logger.info("Loading checkpoint from %s", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)
    logger.info("Checkpoint loaded.")

    # Evaluate the model
    logger.info(
        "Evaluating trained %s checkpoint on %s split", model_cfg["arch"], args.split
    )
    metrics = evaluate_model(
        model=model,
        dataloader=eval_loader,
        device=device,
    )

    # Log metrics to console
    logger.info(
        "Finetuned metrics on %s: "
        f"accuracy={metrics['accuracy'] * 100:.2f}%, "
        f"precision={metrics['precision'] * 100:.2f}%, "
        f"recall={metrics['recall'] * 100:.2f}%, "
        f"f1={metrics['f1'] * 100:.2f}%, "
        f"latency={metrics['latency_ms']:.2f} ms/img, "
        f"throughput={metrics['throughput']:.2f} img/s"
    )

    # Save metrics to disk
    results_dir = Path(cfg["output_dir"]) / "experiments"
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = ckpt_path.stem
    out_path = results_dir / f"{ckpt_name}_{args.split}_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved checkpoint metrics to %s", out_path)


if __name__ == "__main__":
    main()