from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

from src.data.datasets_temp import create_idc_dataloaders
from src.models.model_utils import build_model
from src.training.utils import (
    set_seed,
    train_one_epoch,
    evaluate,
    create_optimizer,
    create_scheduler,
    save_checkpoint,
)
from src.utils.config import load_config, prepare_experiment_dirs
from src.utils.logger import setup_logging, get_logger

from src.evaluation.metrics import evaluate_model

def parse_args():
    parser = argparse.ArgumentParser(description="First results training on IDC dataset")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/first_results_idc_inception_v3.json",
        help="Path to JSON config file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config_path)
    cfg = prepare_experiment_dirs(cfg)

    log_cfg = cfg.get("logging", {})
    setup_logging(
        log_dir=log_cfg.get("log_dir", cfg["output_dir"] + "/logs"),
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

    logger.info("***** Running training *****")
    logger.info(f"  Train examples = {len(train_loader.dataset)}")
    logger.info(f"  Val   examples = {len(val_loader.dataset)}")
    logger.info(f"  Test  examples = {len(test_loader.dataset)}")
    logger.info(f"  Batch size     = {batch_size}")

    model_cfg = cfg["model"]
    model = build_model(
        name=model_cfg["arch"],
        num_classes=int(model_cfg["num_classes"]),
        pretrained=bool(model_cfg.get("pretrained", True)),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {model_cfg['arch']}")
    logger.info(f"  Total params     = {total_params:,}")
    logger.info(f"  Trainable params = {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, cfg["optim"])
    scheduler = create_scheduler(
        optimizer,
        cfg.get("scheduler", {"name": "none"}),
        num_epochs=int(cfg["training"]["num_epochs"]),
    )

    training_cfg = cfg["training"]
    num_epochs = int(training_cfg["num_epochs"])
    use_amp = bool(training_cfg.get("mixed_precision", False)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    logger.info(f"Mixed precision: {use_amp}")

    best_val_acc = 0.0
    output_dir = Path(cfg["output_dir"])

    for epoch in range(1, num_epochs + 1):
        logger.info(f"Epoch {epoch}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
        )

        val_loss, val_acc = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        logger.info(
            f"  Train: loss={train_loss:.4f}, acc={train_acc*100:.2f}% | "
            f"Val: loss={val_loss:.4f}, acc={val_acc*100:.2f}%"
        )

        if scheduler is not None:
            scheduler.step()

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "best_val_acc": best_val_acc,
            "config": cfg,
        }

        save_checkpoint(state, output_dir, filename="last.pt")
        if is_best:
            save_checkpoint(state, output_dir, filename="best.pt")
            logger.info(f"  New best val acc: {best_val_acc*100:.2f}% (checkpoint: best.pt)")

    logger.info("Training finished. Running final test evaluation...")

    test_loss, test_acc = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )
    logger.info(
        f"Test (CE): loss={test_loss:.4f}, acc={test_acc * 100:.2f}%"
    )

    metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
    )
    logger.info(
        "Test metrics: "
        f"accuracy={metrics['accuracy'] * 100:.2f}%, "
        f"precision={metrics['precision'] * 100:.2f}%, "
        f"recall={metrics['recall'] * 100:.2f}%, "
        f"f1={metrics['f1'] * 100:.2f}%, "
        f"latency={metrics['latency_ms']:.2f} ms/img, "
        f"throughput={metrics['throughput']:.2f} img/s"
    )


if __name__ == "__main__":
    main()
