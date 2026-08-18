from pathlib import Path
from PIL import Image
import numpy as np
import json
import time


# Only done for training images.
# Never compute statistics from test data.

file_path = Path(__file__).resolve()
base_path = file_path.parents[2]  # project root


# Processed dataset
processed_data_path = base_path / "processed_data"

processed_rgb_train = processed_data_path / "visible" / "train"
processed_thermal_train = processed_data_path / "infrared" / "train"


STATS_PATH = Path(__file__).parent / "normalization_stats.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def compute_stats(folder_path, mode):

    pixel_sum = None
    pixel_squared_sum = None
    pixel_count = 0

    image_paths = sorted(
        path for path in folder_path.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )

    total = len(image_paths)

    print(f"\nComputing statistics for: {folder_path}")
    print(f"Found {total} images.")

    start_time = time.time()

    for i, image_path in enumerate(image_paths, start=1):

        with Image.open(image_path) as image:
            # with automatically closes the image
            image = image.convert(mode)

            image_array = np.asarray(
                image,
                dtype=np.float64
            ) / 255.0

        if mode == "RGB":
            image_array = image_array.reshape(-1, 3)  # H x W x 3

        else:
            image_array = image_array.reshape(-1, 1)  # H x W x 1

        if pixel_sum is None:
            pixel_sum = np.zeros(image_array.shape[1])
            pixel_squared_sum = np.zeros(image_array.shape[1])

        pixel_sum += image_array.sum(axis=0)
        pixel_squared_sum += (image_array ** 2).sum(axis=0)

        pixel_count += image_array.shape[0]

        # Progress every 500 images
        if i % 500 == 0 or i == total:

            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            remaining = total - i
            eta = remaining / rate if rate > 0 else 0

            print(
                f"\rProcessed: {i}/{total} "
                f"({i / total * 100:.1f}%) | "
                f"Elapsed: {elapsed:.1f}s | "
                f"Rate: {rate:.1f} img/s | "
                f"ETA: {eta:.1f}s",
                end=""
            )

    print()  # move to next line

    mean = pixel_sum / pixel_count

    variance = (
        pixel_squared_sum / pixel_count
    ) - (mean ** 2)

    std = np.sqrt(variance)

    print(f"Mean: {mean}")
    print(f"Std:  {std}")

    return mean, std


def main():

    if STATS_PATH.exists():
        print(f"Statistics file already exists: {STATS_PATH}")
        print("Skipping calculation.")
        return

    print("========================================")
    print(" Computing Dataset Normalization Stats")
    print("========================================")

    print("\nRGB TRAINING DATA")

    rgb_mean, rgb_std = compute_stats(
        processed_rgb_train,
        mode="RGB"
    )

    print("\nTHERMAL TRAINING DATA")

    thermal_mean, thermal_std = compute_stats(
        processed_thermal_train,
        mode="L"
        # L = single-channel grayscale image
    )

    stats = {
        "rgb_mean": rgb_mean.tolist(),
        "rgb_std": rgb_std.tolist(),
        "thermal_mean": thermal_mean.tolist(),
        "thermal_std": thermal_std.tolist()
    }

    with open(STATS_PATH, "w") as file:
        json.dump(stats, file, indent=4)

    print(" Statistics calculation complete!")
    print(f" Saved to: {STATS_PATH}")


if __name__ == "__main__":
    main()