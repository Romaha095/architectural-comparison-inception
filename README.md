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

First experiment configs:

- `configs/first_results_idc_inception_v3.json`
- `configs/first_results_idc_inception_resnet_v2.json`

Key fields:

```json
"data": {
  "root_dir": "data/IDC",
  "img_size": 299,
  "batch_size": 64,
  "num_workers": 4,
  "pin_memory": true,
  "max_images_per_class_train": 5000,
  "max_images_per_class_val": 2000,
  "max_images_per_class_test": 2000
},
"model": {
  "arch": "inception_v3",           // or "inception_resnet_v2"
  "num_classes": 2,
  "pretrained": true
},
"training": {
  "num_epochs": 3,
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

### 6.1 Inception-v3 baseline

```bash
python -m src.evaluation.eval_pretrained_baseline   --config_path configs/first_results_idc_inception_v3.json   --split test
```

### 6.2 Inception-ResNet-v2 baseline

```bash
python -m src.evaluation.eval_pretrained_baseline   --config_path configs/first_results_idc_inception_resnet_v2.json   --split test
```

Results are saved as JSON files under:

```text
results/first_results/...
└── experiments/
    baseline_<arch>_<split>.json
```

---

## 7. Train models on IDC

Training script: `src/training/train_first_results.py`.

### 7.1 Train Inception-v3

```bash
python -m src.training.train_first_results   --config_path configs/first_results_idc_inception_v3.json
```

### 7.2 Train Inception-ResNet-v2

```bash
python -m src.training.train_first_results   --config_path configs/first_results_idc_inception_resnet_v2.json
```

By default, checkpoints are saved to:

```text
results/first_results/inception_v3/
  best.pt
  last.pt

results/first_results/inception_resnet_v2/
  best.pt
  last.pt
```

---

## 8. Evaluate trained checkpoints

These commands load a saved checkpoint and evaluate it on a chosen split using `metrics.py`.

### 8.1 Inception-v3 (best checkpoint on test split)

```bash
python -m src.evaluation.eval_trained_checkpoint   --config_path configs/first_results_idc_inception_v3.json   --checkpoint_path results/first_results/inception_v3/best.pt   --split test
```

### 8.2 Inception-ResNet-v2 (best checkpoint on test split)

```bash
python -m src.evaluation.eval_trained_checkpoint   --config_path configs/first_results_idc_inception_resnet_v2.json   --checkpoint_path results/first_results/inception_resnet_v2/best.pt   --split test
```

Metrics are saved to:

```text
results/first_results/...
└── experiments/
    <checkpoint_name>_<split>_metrics.json
```

Make sure you always run commands from the **project root** and with the virtual environment activated.
