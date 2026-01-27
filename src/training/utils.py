from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from tqdm.auto import tqdm

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _split_inception_outputs(
    outputs: Any,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    aux_logits = None
    if hasattr(outputs, "logits") and hasattr(outputs, "aux_logits"):
        logits = outputs.logits
        aux_logits = outputs.aux_logits
    elif isinstance(outputs, (tuple, list)) and len(outputs) == 2:
        logits, aux_logits = outputs
    else:
        logits = outputs
    return logits, aux_logits

def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = False,
    aux_loss_weight: float = 0.4,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    dataset_size = len(dataloader.dataset)

    pbar = tqdm(
        total=dataset_size,
        desc="Train",
        unit="img",
        leave=True,
    )

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with autocast():
                outputs = model(inputs)
                logits, aux_logits = _split_inception_outputs(outputs)
                loss = criterion(logits, targets)
                if aux_logits is not None:
                    loss = loss + aux_loss_weight * criterion(aux_logits, targets)
        else:
            outputs = model(inputs)
            logits, aux_logits = _split_inception_outputs(outputs)
            loss = criterion(logits, targets)
            if aux_logits is not None:
                loss = loss + aux_loss_weight * criterion(aux_logits, targets)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

        avg_loss = total_loss / max(total_samples, 1)
        avg_acc = total_correct / max(total_samples, 1)

        pbar.update(batch_size)
        pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "acc": f"{avg_acc * 100:.2f}%",
            "seen": total_samples,
        })

    pbar.close()

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    aux_loss_weight: float = 0.4,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            logits, aux_logits = _split_inception_outputs(outputs)
            loss = criterion(logits, targets)
            if aux_logits is not None:
                loss = loss + aux_loss_weight * criterion(aux_logits, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


def create_optimizer(
    model: nn.Module,
    optim_cfg: Dict[str, Any],
) -> optim.Optimizer:
    name = optim_cfg.get("name", "adam").lower()
    lr = float(optim_cfg.get("lr", 3e-4))
    lr_backbone = float(optim_cfg.get("lr_backbone", lr))
    lr_head = float(optim_cfg.get("lr_head", lr))
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))

    head_params = []
    backbone_params = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("fc.") or n.startswith("AuxLogits.fc."):
            head_params.append(p)
        else:
            backbone_params.append(p)

    if len(head_params) == 0:
        params = [p for p in model.parameters() if p.requires_grad]
        if name == "adam":
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)
        if name == "adamw":
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        if name == "sgd":
            momentum = float(optim_cfg.get("momentum", 0.9))
            nesterov = bool(optim_cfg.get("nesterov", True))
            return optim.SGD(
                params, lr=lr, weight_decay=weight_decay,
                momentum=momentum, nesterov=nesterov
            )
        raise ValueError(f"Unknown optimizer: {name}")

    param_groups = [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ]

    if name == "adam":
        return optim.Adam(param_groups, lr=lr_head, weight_decay=weight_decay)
    if name == "adamw":
        return optim.AdamW(param_groups, lr=lr_head, weight_decay=weight_decay)
    if name == "sgd":
        momentum = float(optim_cfg.get("momentum", 0.9))
        nesterov = bool(optim_cfg.get("nesterov", True))
        return optim.SGD(
            param_groups,
            lr=lr_head,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
        )

    raise ValueError(f"Unknown optimizer: {name}")


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_cfg: Dict[str, Any],
    num_epochs: int,
) -> Optional[optim.lr_scheduler._LRScheduler]:
    name = scheduler_cfg.get("name", "none").lower()
    if name == "none":
        return None

    if name == "step":
        step_size = int(scheduler_cfg.get("step_size", max(1, num_epochs // 3)))
        gamma = float(scheduler_cfg.get("gamma", 0.1))
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    raise ValueError(f"Unknown scheduler: {name}")


def save_checkpoint(
    state: Dict[str, Any],
    output_dir: str | Path,
    filename: str = "checkpoint.pt",
) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / filename
    torch.save(state, ckpt_path)
    return str(ckpt_path)
