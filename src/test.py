import os
import torch
import json
from PIL import Image
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from models.generator import Generator
from datasets.transform import PairedTransform, ToTensor, Normalize
from datasets.dataset import LLVIPDataset

def run_evaluation(checkpoint_path="saved_models/generator_epoch_100.pth", num_samples=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "eval_results"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Normalization Stats
    with open("normalization_stats.json", "r") as f:
        stats = json.load(f)

    rgb_mean = stats["rgb_mean"]
    rgb_std = stats["rgb_std"]
    thermal_mean = stats["thermal_mean"]
    thermal_std = stats["thermal_std"]

    transform = PairedTransform([
        ToTensor(),
        Normalize(
            rgb_mean=rgb_mean,
            rgb_std=rgb_std,
            thermal_mean=thermal_mean,
            thermal_std=thermal_std
        )
    ])

    # 2. Load Test Dataset
    test_dataset = LLVIPDataset(train=False, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=num_samples, shuffle=True)

    # 3. Load Model
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generator.eval()

    # 4. Generate & Save Side-by-Side Visual Comparisons
    with torch.no_grad():
        rgb, thermal_real = next(iter(test_dataloader))
        rgb = rgb.to(device)
        thermal_real = thermal_real.to(device)

        thermal_fake = generator(rgb)

        # Un-normalize for saving real-looking images
        for c in range(3):
            rgb[:, c] = rgb[:, c] * rgb_std[c] + rgb_mean[c]

        thermal_real = thermal_real * thermal_std[0] + thermal_mean[0]
        thermal_fake = thermal_fake * thermal_std[0] + thermal_mean[0]

        # Clamp values to valid [0, 1] range
        rgb = torch.clamp(rgb, 0.0, 1.0)
        thermal_real = torch.clamp(thermal_real, 0.0, 1.0)
        thermal_fake = torch.clamp(thermal_fake, 0.0, 1.0)

        # Save individual comparison triplets: [RGB Input | Real Thermal | Generated Thermal]
        for i in range(min(num_samples, len(rgb))):
            # Expand single channel thermal to 3 channels for concatenated side-by-side saving
            t_real_3ch = thermal_real[i].repeat(3, 1, 1)
            t_fake_3ch = thermal_fake[i].repeat(3, 1, 1)

            comparison = torch.cat([rgb[i], t_real_3ch, t_fake_3ch], dim=2)
            save_image(comparison, f"{output_dir}/sample_{i+1}.png")

    print(f"Saved {num_samples} visual comparisons to '{output_dir}/'")

if __name__ == "__main__":
    run_evaluation()