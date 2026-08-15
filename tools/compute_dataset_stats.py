from pathlib import Path
from PIL import Image
import numpy as np

# only done for training images. never done for test.
# FIX HERE: point these to the RESIZED training folders.
RGB_TRAIN_PATH = Path(r"...")
THERMAL_TRAIN_PATH = Path(r"...")


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def compute_stats(folder_path, mode):

    pixel_sum = None
    pixel_squared_sum = None
    pixel_count = 0

    image_paths = sorted(
        path for path in folder_path.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )

    for image_path in image_paths:

        with Image.open(image_path) as image:
        # with directly closes the image automatically
            image = image.convert(mode)

            image_array = np.asarray(image, dtype=np.float64) / 255.0

        if mode == "RGB":
            image_array = image_array.reshape(-1, 3) # h x w x 3 

        else:
            image_array = image_array.reshape(-1, 1) # h x w

        if pixel_sum is None:
            pixel_sum = np.zeros(image_array.shape[1])
            pixel_squared_sum = np.zeros(image_array.shape[1])

        pixel_sum += image_array.sum(axis=0)
        pixel_squared_sum += (image_array ** 2).sum(axis=0)
        pixel_count += image_array.shape[0]

    mean = pixel_sum / pixel_count
    variance = (pixel_squared_sum / pixel_count) - (mean ** 2)
    std = np.sqrt(variance)

    return mean, std


def main():

    rgb_mean, rgb_std = compute_stats(
        RGB_TRAIN_PATH,
        mode="RGB"
    )

    thermal_mean, thermal_std = compute_stats(
        THERMAL_TRAIN_PATH,
        mode="L"
        # mode L means single channel grayscale image (pil)
    )

    print("RGB statistics:")
    print("Mean:", rgb_mean)
    print("Std: ", rgb_std)

    print()

    print("Thermal statistics:")
    print("Mean:", thermal_mean)
    print("Std: ", thermal_std)


if __name__ == "__main__":
    main()