import json
from pathlib import Path

import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from models.generator import Generator


# FIX HERE: path to the test RGB image
RGB_IMAGE_PATH = Path(r"...")

# FIX HERE: path to the corresponding real thermal image
THERMAL_IMAGE_PATH = Path(r"...")

# FIX HERE: path to the trained Generator checkpoint
GENERATOR_WEIGHTS_PATH = Path(r"...")

# FIX HERE: path to normalization_stats.json
STATS_PATH = Path(r"...")


IMAGE_SIZE = (256, 256)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_normalization_stats():

    with open(STATS_PATH, "r") as file:
        stats = json.load(file)

    rgb_mean = np.array(stats["rgb_mean"])
    rgb_std = np.array(stats["rgb_std"])

    return rgb_mean, rgb_std


def load_rgb_image(image_path, rgb_mean, rgb_std):

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    # H x W x 3 → 3 x H x W
    image_tensor = torch.from_numpy(
        image_array
    ).permute(2, 0, 1)

    # Normalize using the same statistics used during training.
    mean = torch.tensor(
        rgb_mean,
        dtype=torch.float32
    ).view(3, 1, 1)

    std = torch.tensor(
        rgb_std,
        dtype=torch.float32
    ).view(3, 1, 1)

    image_tensor = (image_tensor - mean) / std

    # Add batch dimension.
    return image_tensor.unsqueeze(0)


def main():

    # Load the normalization values calculated from
    # the training dataset.
    rgb_mean, rgb_std = load_normalization_stats()

    # Build Generator.
    generator = Generator().to(DEVICE)

    # Load trained Generator weights.
    generator.load_state_dict(
        torch.load(
            GENERATOR_WEIGHTS_PATH,
            map_location=DEVICE
        )
    )

    # Evaluation mode.
    generator.eval()

    # Load and normalize test RGB image.
    rgb_tensor = load_rgb_image(
        RGB_IMAGE_PATH,
        rgb_mean,
        rgb_std
    ).to(DEVICE)

    # Generate thermal image.
    with torch.no_grad():
        fake_thermal = generator(rgb_tensor)

    # Convert generated output from [-1, 1] to [0, 1]
    # for displaying.
    fake_thermal = fake_thermal.squeeze(0).cpu()

    fake_thermal = (fake_thermal + 1) / 2
    fake_thermal = fake_thermal.clamp(0, 1)

    fake_thermal = fake_thermal.squeeze(0).numpy()

    # Load real thermal image.
    real_thermal = Image.open(
        THERMAL_IMAGE_PATH
    ).convert("L")

    real_thermal = real_thermal.resize(IMAGE_SIZE)

    # Load RGB image only for display.
    rgb_display = Image.open(
        RGB_IMAGE_PATH
    ).convert("RGB")

    rgb_display = rgb_display.resize(IMAGE_SIZE)

    # Display the three images.
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(rgb_display)
    plt.title("RGB Input")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(fake_thermal, cmap="gray")
    plt.title("Generated Thermal")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(real_thermal, cmap="gray")
    plt.title("Real Thermal")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()