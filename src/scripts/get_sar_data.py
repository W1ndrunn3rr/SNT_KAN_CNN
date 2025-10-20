import kagglehub
import shutil
import os
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split

print(f"Downloading dataset from Kaggle...")

source_path = kagglehub.dataset_download(
    "requiemonk/sentinel12-image-pairs-segregated-by-terrain",
)

print(f"\nDataset downloaded to: {source_path}")

# Create target directories
sar_1_dir = Path("sar_1")
sar_2_dir = Path("sar_2")

sar_1_train = sar_1_dir / "train"
sar_1_test = sar_1_dir / "test"
sar_2_train = sar_2_dir / "train"
sar_2_test = sar_2_dir / "test"

for dir_path in [sar_1_train, sar_1_test, sar_2_train, sar_2_test]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Navigate to the versioned data folder
source_path_obj = Path(source_path) / "v_2"

# Get all class directories (agri, barren, etc.)
class_folders = [f for f in source_path_obj.iterdir() if f.is_dir()]

print(f"\nProcessing terrain classes...")

for class_folder in class_folders:
    class_name = class_folder.name
    print(f"Processing class: {class_name}")

    # Get s1 and s2 subdirectories
    s1_folder = class_folder / "s1"
    s2_folder = class_folder / "s2"

    if not s1_folder.exists() or not s2_folder.exists():
        print(f"  Warning: Missing s1 or s2 folder in {class_name}")
        continue

    # Get all files from s1 and s2
    sar_1_files = sorted(glob.glob(str(s1_folder / "*.png")))
    sar_2_files = sorted(glob.glob(str(s2_folder / "*.png")))

    if not sar_1_files or not sar_2_files:
        print(f"  Warning: Missing files in {class_name}")
        continue

    # Split files into train/test (80/20)
    sar_1_train_files, sar_1_test_files = train_test_split(
        sar_1_files, test_size=0.2, random_state=42
    )
    sar_2_train_files, sar_2_test_files = train_test_split(
        sar_2_files, test_size=0.2, random_state=42
    )

    # Create class directories
    sar_1_train_class = sar_1_train / class_name
    sar_1_test_class = sar_1_test / class_name
    sar_2_train_class = sar_2_train / class_name
    sar_2_test_class = sar_2_test / class_name

    for dir_path in [
        sar_1_train_class,
        sar_1_test_class,
        sar_2_train_class,
        sar_2_test_class,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Copy SAR-1 files
    for file in sar_1_train_files:
        shutil.copy2(file, sar_1_train_class / Path(file).name)
    for file in sar_1_test_files:
        shutil.copy2(file, sar_1_test_class / Path(file).name)

    # Copy SAR-2 files
    for file in sar_2_train_files:
        shutil.copy2(file, sar_2_train_class / Path(file).name)
    for file in sar_2_test_files:
        shutil.copy2(file, sar_2_test_class / Path(file).name)

    print(f"  SAR-1: {len(sar_1_train_files)} train, {len(sar_1_test_files)} test")
    print(f"  SAR-2: {len(sar_2_train_files)} train, {len(sar_2_test_files)} test")

print(f"\nDatasets ready:")
print(f"  SAR-1: {sar_1_dir.absolute()}")
print(f"  SAR-2: {sar_2_dir.absolute()}")
print(f"\nOriginal data: {source_path}")
