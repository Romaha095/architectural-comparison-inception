from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.datasets_temp import create_idc_dataloaders
from src.evaluation.metrics import evaluate_model
from src.models.model_utils import build_model
from src.training.utils import set_seed
from src.utils.config import load_config, prepare_experiment_dirs
from src.utils.logger import setup_logging, get_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate pure ImageNet-pretrained model as baseline on IDC dataset"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/first_results_idc_inception_v3.json",
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate on",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config_path)
    cfg = prepare_experiment_dirs(cfg)

    log_cfg = cfg.get("logging", {})
    setup_logging(
        log_dir=log_cfg.get("log_dir", Path(cfg["output_dir"]) / "logs"),
        log_level=log_cfg.get("log_level", "INFO"),
        log_to_file=log_cfg.get("log_to_file", True),
    )
    logger = get_logger(__name__)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    logger.info(f"Seed set to {seed}")

    device_str = cfg.get("training", {}).get(
        "device",
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    data_cfg = cfg["data"]
    root_dir = data_cfg["root_dir"]
    img_size = int(data_cfg.get("img_size", 299))
    batch_size = int(data_cfg.get("batch_size", 64))
    num_workers = int(data_cfg.get("num_workers", 4))
    pin_memory = bool(data_cfg.get("pin_memory", True))

    max_train = data_cfg.get("max_images_per_class_train")
    max_val = data_cfg.get("max_images_per_class_val")
    max_test = data_cfg.get("max_images_per_class_test")

    train_loader, val_loader, test_loader = create_idc_dataloaders(
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

    logger.info("Building pure pretrained model (no IDC fine-tuning)...")
    model_cfg = cfg["model"]
    model = build_model(
        name=model_cfg["arch"],
        num_classes=int(model_cfg["num_classes"]),
        pretrained=True,
        freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
    )
    model.to(device)

    logger.info(
        f"Evaluating {model_cfg['arch']} (ImageNet-pretrained, no IDC training) on {args.split} split"
    )

    metrics = evaluate_model(
        model=model,
        dataloader=eval_loader,
        device=device,
    )


    logger.info(
        "Baseline metrics on %s: ",
        f"accuracy={metrics['accuracy'] * 100:.2f}%, "
        f"precision={metrics['precision'] * 100:.2f}%, "
        f"recall={metrics['recall'] * 100:.2f}%, "
        f"f1={metrics['f1'] * 100:.2f}%, "
        f"latency={metrics['latency_ms']:.2f} ms/img, "
        f"throughput={metrics['throughput']:.2f} img/s"
    )

    results_dir = Path(cfg["output_dir"]) / "experiments"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"baseline_{model_cfg['arch']}_{args.split}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved baseline metrics to %s", out_path)


if __name__ == "__main__":
    main()