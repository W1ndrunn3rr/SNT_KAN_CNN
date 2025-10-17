import os
import glob
import shutil
from sklearn.model_selection import train_test_split

base_dir = "data"

subfolders = [f.path for f in os.scandir(base_dir) if f.is_dir()]

for folder in subfolders:
    files_to_delete = glob.glob(os.path.join(folder, "*.Identifier"))
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"Removed: {file_path}")
        except OSError as e:
            print(f"Error removing {file_path}: {e}")

train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

for folder in subfolders:
    class_name = os.path.basename(folder)
    files = glob.glob(os.path.join(folder, "*"))

    if not files:
        print(f"No files to split in {folder}")
        continue

    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)

    train_class_dir = os.path.join(train_dir, class_name)
    test_class_dir = os.path.join(test_dir, class_name)

    if not os.path.exists(train_class_dir):
        os.makedirs(train_class_dir)
    if not os.path.exists(test_class_dir):
        os.makedirs(test_class_dir)

    for file in train_files:
        shutil.move(file, os.path.join(train_class_dir, os.path.basename(file)))

    for file in test_files:
        shutil.move(file, os.path.join(test_class_dir, os.path.basename(file)))

    print(f"Moved files for class {class_name} to train and test directories.")

print("\nData preparation complete.")
