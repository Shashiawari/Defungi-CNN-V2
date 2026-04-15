import argparse
import os
import shutil
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and extract an image dataset from Kaggle.")
    parser.add_argument(
        "--dataset",
        default="anshtanwar/microscopic-fungi-images",
        help="Dataset slug in the form owner/dataset-name.",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset",
        help="Directory where the downloaded dataset will be extracted.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the output directory if it already exists.",
    )
    return parser.parse_args()


def ensure_clean_dir(path: Path, force: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not force:
            raise FileExistsError(
                f"{path} already exists and is not empty. Use --force to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    archive_dir = output_dir.parent / "_kaggle_download"

    ensure_clean_dir(output_dir, args.force)
    archive_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading dataset {args.dataset}...")
    api.dataset_download_files(args.dataset, path=str(archive_dir), unzip=False)

    zip_files = sorted(archive_dir.glob("*.zip"), key=os.path.getmtime, reverse=True)
    if not zip_files:
        raise FileNotFoundError("Kaggle download finished, but no zip archive was found.")

    archive_path = zip_files[0]
    print(f"Extracting {archive_path.name} into {output_dir}...")

    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(output_dir)

    print(f"Dataset ready at {output_dir}")


if __name__ == "__main__":
    main()
