from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

from src.data.datasets_temp import create_idc_dataloaders
from src.models.inception_resnet_v2 import build_inception_resnet_v2
from src.training.utils import (
    set_seed,
    train_one_epoch,
    evaluate,
    save_checkpoint,
)
from src.utils.config import load_config, prepare_experiment_dirs
from src.utils.logger import setup_logging, get_logger
from src.evaluation.metrics import evaluate_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Two-stage training for Inception-ResNet-v2 on IDC")
    p.add_argument("--config_path", type=str, required=True)
    return p.parse_args()


def freeze_all(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_named_children(model: torch.nn.Module, names: set[str]) -> set[str]:
    found = set()
    for name, module in model.named_children():
        if name in names:
            for p in module.parameters():
                p.requires_grad = True
            found.add(name)
    return found


def make_optimizer(model: torch.nn.Module, optim_cfg: dict) -> torch.optim.Optimizer:
    name = str(optim_cfg.get("name", "adam")).lower()
    lr = float(optim_cfg.get("lr", 3e-4))
    wd = float(optim_cfg.get("weight_decay", 0.0))

    params = [p for p in model.parameters() if p.requires_grad]  # ключевой момент
    if name == "adam" or name == "adamw":
        # в вашем проекте был "adam" — оставляем Adam
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        momentum = float(optim_cfg.get("momentum", 0.9))
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
    raise ValueError(f"Unsupported optimizer: {name}")


def main() -> None:
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

    training_cfg = cfg.get("training", {})
    device_str = training_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    # data
    data_cfg = cfg["data"]
    train_loader, val_loader, test_loader = create_idc_dataloaders(
        data_root=data_cfg["root_dir"],
        img_size=int(data_cfg.get("img_size", 299)),
        batch_size=int(data_cfg.get("batch_size", 64)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        max_images_per_class_train=data_cfg.get("max_images_per_class_train"),
        max_images_per_class_val=data_cfg.get("max_images_per_class_val"),
        max_images_per_class_test=data_cfg.get("max_images_per_class_test"),
        seed=seed,
        pin_memory=bool(data_cfg.get("pin_memory", True)),
    )

    logger.info("***** Running training *****")
    logger.info(f"  Train examples = {len(train_loader.dataset)}")
    logger.info(f"  Val   examples = {len(val_loader.dataset)}")
    logger.info(f"  Test  examples = {len(test_loader.dataset)}")
    logger.info(f"  Batch size     = {data_cfg.get('batch_size', 64)}")

    # model
    model_cfg = cfg["model"]
    model = build_inception_resnet_v2(
        num_classes=int(model_cfg["num_classes"]),
        pretrained=bool(model_cfg.get("pretrained", True)),
        freeze_backbone=False,  # будем управлять заморозкой сами
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    use_amp = bool(training_cfg.get("mixed_precision", False)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    logger.info(f"Mixed precision: {use_amp}")

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Two stages config ----
    stage1_epochs = int(training_cfg.get("stage1_epochs", 10))
    stage2_epochs = int(training_cfg.get("stage2_epochs", 10))
    stage2_lr_mult = float(training_cfg.get("stage2_lr_mult", 0.1))
    stage2_train_classifier = bool(training_cfg.get("stage2_train_classifier", True))

    # =========================
    # STAGE 1: classifier only
    # =========================
    logger.info(f"===== STAGE 1: train classifier only for {stage1_epochs} epochs =====")
    freeze_all(model)
    found = unfreeze_named_children(model, {"classif"})
    if "classif" not in found:
        logger.warning("classif not found in model.named_children()")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: total={total_params:,} trainable={trainable_params:,}")

    optim_cfg1 = deepcopy(cfg["optim"])
    optimizer = make_optimizer(model, optim_cfg1)

    best_val_acc = 0.0
    best_state_stage1 = None

    for epoch in range(1, stage1_epochs + 1):
        logger.info(f"Stage1 Epoch {epoch}/{stage1_epochs}")

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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_stage1 = deepcopy(model.state_dict())
            logger.info(f"  New best stage1 val acc: {best_val_acc*100:.2f}%")

        # чекпоинты stage1
        state = {
            "stage": 1,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "config": cfg,
        }
        save_checkpoint(state, out_dir, filename="last_stage1.pt")
        if val_acc >= best_val_acc:
            save_checkpoint(state, out_dir, filename="best_stage1.pt")

    # поднимаем лучший stage1 перед stage2
    if best_state_stage1 is not None:
        model.load_state_dict(best_state_stage1)
        logger.info("Loaded best Stage1 weights into model for Stage2.")

    # =========================
    # STAGE 2: unfreeze selected backbone blocks
    # =========================
    logger.info(
        f"===== STAGE 2: train repeat_2 + block8 + conv2d_7b "
        f"{'+ classif' if stage2_train_classifier else ''} for {stage2_epochs} epochs ====="
    )

    freeze_all(model)
    blocks = {"repeat_2", "block8", "conv2d_7b"}
    if stage2_train_classifier:
        blocks.add("classif")

    found2 = unfreeze_named_children(model, blocks)
    missing2 = blocks - found2
    if missing2:
        logger.warning(f"Stage2 requested but not found: {sorted(missing2)}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: total={total_params:,} trainable={trainable_params:,}")

    # новый optimizer с меньшим lr (обычно так делают для backbone)
    optim_cfg2 = deepcopy(cfg["optim"])
    optim_cfg2["lr"] = float(optim_cfg2.get("lr", 3e-4)) * stage2_lr_mult
    optimizer = make_optimizer(model, optim_cfg2)

    best_val_acc2 = 0.0
    best_state_stage2 = None

    for epoch in range(1, stage2_epochs + 1):
        logger.info(f"Stage2 Epoch {epoch}/{stage2_epochs}")

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

        if val_acc > best_val_acc2:
            best_val_acc2 = val_acc
            best_state_stage2 = deepcopy(model.state_dict())
            logger.info(f"  New best stage2 val acc: {best_val_acc2*100:.2f}%")

        state = {
            "stage": 2,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_acc": best_val_acc2,
            "config": cfg,
        }
        save_checkpoint(state, out_dir, filename="last.pt")
        if val_acc >= best_val_acc2:
            save_checkpoint(state, out_dir, filename="best.pt")

    if best_state_stage2 is not None:
        model.load_state_dict(best_state_stage2)

    logger.info("Training finished. Running final test evaluation...")

    test_loss, test_acc = evaluate(model=model, dataloader=test_loader, criterion=criterion, device=device)
    logger.info(f"Test (CE): loss={test_loss:.4f}, acc={test_acc*100:.2f}%")

    metrics = evaluate_model(model=model, dataloader=test_loader, device=device)
    logger.info(
        "Test metrics: "
        f"accuracy={metrics['accuracy']*100:.2f}%, "
        f"precision={metrics['precision']*100:.2f}%, "
        f"recall={metrics['recall']*100:.2f}%, "
        f"f1={metrics['f1']*100:.2f}%, "
        f"latency={metrics['latency_ms']:.2f} ms/img, "
        f"throughput={metrics['throughput']:.2f} img/s"
    )


if __name__ == "__main__":
    main()
