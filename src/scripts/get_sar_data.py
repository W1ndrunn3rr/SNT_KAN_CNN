import kagglehub
import shutil
import os
from pathlib import Path

TARGET_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

print(f"🔄 Downloading dataset from Kaggle...")

source_path = kagglehub.dataset_download(
    "requiemonk/sentinel12-image-pairs-segregated-by-terrain",
)

print(f"✅ Dataset downloaded to cache: {source_path}")
print(f"📦 Copying to project directory: {TARGET_DIR}")

for item in Path(source_path).iterdir():
    dest = TARGET_DIR / item.name
    if item.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(item, dest)
        print(f"   📁 Copied directory: {item.name}")
    else:
        shutil.copy2(item, dest)
        print(f"   📄 Copied file: {item.name}")

print(f"\n✅ Dataset ready at: {TARGET_DIR.absolute()}")
