# Architectural Comparison: Inception-v3 vs. Inception-ResNet-v2

This repository compares **Inception-v3** and **Inception-ResNet-v2** on the IDC breast cancer histopathology dataset (ImageNet-pretrained backbones, fine-tuned on IDC).

---

## 1. Clone the repository

```bash
git clone 
cd architectural-comparison-inception
```

---

## 2. Create and activate virtual environment

### 2.1 Create venv (Windows / Linux / macOS)

```bash
python -m venv .venv
```

### 2.2 Activate venv

**Windows (PowerShell)**

```powershell
.\.venv\Scripts\activate
```

**Linux / macOS (bash/zsh)**

```bash
source .venv/bin/activate
```

---

## 3. Install dependencies

### 3.1 Install PyTorch with CUDA 12.6

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

(If you’re on CPU-only, install the appropriate CPU wheel instead.)

### 3.2 Install project requirements

```bash
pip install -r requirements.txt
```

---

## 4. Download the IDC dataset

The project uses a helper script that downloads and unpacks the IDC dataset into `data/IDC`.

Run (from the project root):

```bash
python -m src.data.download_dataset
```

After this, the dataset structure should look like:

```text
data/
  IDC/
    training/
      0/
      1/
    validation/
      0/
      1/
    testing/
      0/
      1/
```

---

## 5. Configuration files

Еxperiment config:

- `configs/train_inception_resnet_v2.json`
- `first_results_idc_inception_resnet_v2_full.json`
Key fields:

```json
"data": {
  "root_dir": "data/IDC",
  "img_size": 299,
  "batch_size": 64,
  "num_workers": 4,
  "pin_memory": true,
  "max_images_per_class_train": null,
  "max_images_per_class_val": null,
  "max_images_per_class_test": null
},
"model": {
  "arch": "inception_resnet_v2",
  "num_classes": 2,
  "pretrained": true
},
"training": {
  "num_epochs": 10,
  "device": "cuda",
  "mixed_precision": true
}
```

- `root_dir` — folder containing `training/`, `validation/`, `testing`.
- `max_images_per_class_*` — per-class caps for quick experiments.  
  - Set them to a larger value to use more data.
  - Set them to `null` or remove these keys to use **all images** in each split.

---

## 6. Evaluate pretrained baseline (no IDC fine-tuning)

These scripts evaluate a **pure ImageNet-pretrained** model using `metrics.py`  
(accuracy, precision, recall, F1, latency, throughput).

> Run all commands from the project root with the venv active.

### 6.1 Inception-ResNet-v2 baseline

```bash
python -m src.evaluation.eval_pretrained_baseline     --config_path configs/first_results_idc_inception_resnet_v2_full.json     --split test
```

Results are saved as JSON files under:

```text
results/first_results/...
└── experiments/
    baseline_<arch>_<split>.json
```

---

## 7. Train models on IDC

Training script: `src/training/train_inception_resnet_v2.py`.

### 7.1 Train Inception-ResNet-v2

```bash
python -m src.training.train_inception_resnet_v2     --config_path configs/train_inception_resnet_v2.json
```

Сheckpoints are saved to:

```text

results/train_inception_resnet_v2/
  best.pt
  last.pt
```

---

## 8. Evaluate trained checkpoints

These commands load a saved checkpoint and evaluate it on a chosen split using `metrics.py`.

### 8.1 Inception-ResNet-v2 (best checkpoint on test split)

```bash
python -m src.evaluation.eval_trained_checkpoint     --config_path configs/train_inception_resnet_v2.json     --checkpoint_path results/train_inception_resnet_v2/best.pt     --split test
```

Metrics are saved to:

```text
results/train_inception_resnet_v2/...
└── experiments/
    <checkpoint_name>_<split>_metrics.json
```

Make sure you always run commands from the **project root** and with the virtual environment activated.
