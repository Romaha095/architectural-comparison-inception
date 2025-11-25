import time
from pathlib import Path
import zipfile
import requests
from tqdm import tqdm


KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/kasikrit/idc-dataset"


def download_file(url: str, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))

        chunk_size = 1024 * 1024  # 1 MB
        progress = tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {dst_path.name}",
        )

        start_time = time.time()
        with dst_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))

        progress.close()
        elapsed = time.time() - start_time
        if total > 0:
            size_mb = total / (1024 ** 2)
            speed = size_mb / elapsed if elapsed > 0 else 0.0
            print(f"\nDownloaded {size_mb:.2f} MB in {elapsed:.1f} s ({speed:.2f} MB/s).")
        else:
            print(f"\nDownload finished in {elapsed:.1f} s.")


def unzip_file(zip_path: Path, extract_to: Path) -> None:
    print(f"Unzipping {zip_path.name} to {extract_to} ...")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print("Unzip done.")


def main():
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data"
    zip_path = data_dir / "idc-dataset.zip"
    extract_dir = data_dir

    if zip_path.exists():
        print(f"{zip_path} already exists, skipping download.")
    else:
        print(f"Downloading IDC dataset to {zip_path} ...")
        download_file(KAGGLE_URL, zip_path)

    unzip_file(zip_path, extract_dir)

    print("\nDataset is ready.")
    print(f"Root dir for config: {data_dir / 'IDC'}")


if __name__ == "__main__":
    main()
