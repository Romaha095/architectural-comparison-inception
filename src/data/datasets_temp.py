from pathlib import Path
from typing import Sequence, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder

from .transforms import get_transforms


class SubsetWithTransform(Dataset):
    def __init__(self, base_dataset: ImageFolder, indices: Sequence[int], transform=None):
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

    by_class = {}
    for idx, (_, target) in enumerate(base_dataset.samples):
        by_class.setdefault(target, []).append(idx)

    g = torch.Generator().manual_seed(seed)
    selected = []
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


def create_idc_datasets(
    data_root: str | Path,
    img_size: int = 299,
    max_images_per_class_train: Optional[int] = 5000,
    max_images_per_class_val: Optional[int] = 2000,
    max_images_per_class_test: Optional[int] = 2000,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    data_root = Path(data_root)
    train_dir = data_root / "training"
    val_dir = data_root / "validation"
    test_dir = data_root / "testing"

    if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Expected training/validation/testing under {data_root}")

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


def create_idc_dataloaders(
    data_root: str | Path,
    img_size: int = 299,
    batch_size: int = 64,
    num_workers: int = 4,
    max_images_per_class_train: Optional[int] = 5000,
    max_images_per_class_val: Optional[int] = 2000,
    max_images_per_class_test: Optional[int] = 2000,
    seed: int = 42,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds, val_ds, test_ds = create_idc_datasets(
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
