import os
from PIL import Image
from pathlib import Path
import io


def check_and_remove_corrupted_images(data_dir):
    corrupted_files = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                filepath = os.path.join(root, file)
                try:
                    # Try to open and fully load the image
                    with open(filepath, "rb") as f:
                        img = Image.open(f)
                        img.load()  # Force loading the image data
                        # Try to convert to RGB (common operation in data loading)
                        img.convert("RGB")
                except Exception as e:
                    print(f"Corrupted image found: {filepath}")
                    print(f"Error: {e}")
                    corrupted_files.append(filepath)

    if corrupted_files:
        print(f"\nFound {len(corrupted_files)} corrupted files.")
        response = input("Do you want to delete them? (yes/no): ")
        if response.lower() == "yes":
            for file in corrupted_files:
                os.remove(file)
                print(f"Deleted: {file}")
    else:
        print("No corrupted images found.")


# Also check the specific problematic file
def check_specific_file():
    filepath = "/home/nixos/SNT_KAN_CNN/data/drones/train/2_blade_rotor/figure1.jpg"
    print(f"\nChecking specific file: {filepath}")
    print(f"File exists: {os.path.exists(filepath)}")
    print(
        f"File size: {os.path.getsize(filepath) if os.path.exists(filepath) else 'N/A'}"
    )

    try:
        with open(filepath, "rb") as f:
            img = Image.open(f)
            print(f"Image format: {img.format}")
            print(f"Image size: {img.size}")
            print(f"Image mode: {img.mode}")
            img.load()
            print("Successfully loaded image")
    except Exception as e:
        print(f"Error loading image: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    data_dir = "/home/nixos/SNT_KAN_CNN/data/drones"
    check_specific_file()
    print("\n" + "=" * 50 + "\n")
    check_and_remove_corrupted_images(data_dir)
