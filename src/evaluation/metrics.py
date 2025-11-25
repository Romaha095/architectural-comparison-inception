from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    latencies: List[float] = []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.size(0)

        start_time = time.time()
        outputs = model(images)
        end_time = time.time()

        if hasattr(outputs, "logits"):
            outputs = outputs.logits
        elif isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        batch_latency = (end_time - start_time) / batch_size
        latencies.append(batch_latency)

        preds = outputs.argmax(dim=1)
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    latency_ms = float(np.mean(latencies)) * 1000.0
    throughput = 1.0 / float(np.mean(latencies))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "latency_ms": latency_ms,
        "throughput": throughput,
    }
