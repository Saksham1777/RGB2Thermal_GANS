import os
import shutil
import argparse
import torch
from torchvision.utils import save_image

from models.generator import Generator
from datasets.dataset import LLVIPDataset
from datasets.transform import PairedTransform, ToTensor, Normalize


def parse_args():
    parser = argparse.ArgumentParser(description="Save model milestone and sample comparisons.")
    parser.add_argument("--epoch", type=int, required=True, help="Current epoch number to label (e.g., --epoch 10)")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of test samples to generate")
    return parser.parse_args()


def main():
    args = parse_args()
    epoch_num = args.epoch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    ckpt_latest = "saved_models/checkpoint_latest.pth"
    milestone_ckpt = f"saved_models/checkpoint_epoch{epoch_num}.pth"
    output_dir = f"milestone_outputs/epoch_{epoch_num}"
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(ckpt_latest):
        print(f"Error: Could not find {ckpt_latest}. Make sure training has started/saved.")
        return

    # 1. Archive Checkpoint
    shutil.copyfile(ckpt_latest, milestone_ckpt)
    print(f"[✓] Archived checkpoint to: {milestone_ckpt}")

    # 2. Initialize Model
    generator = Generator().to(device)
    checkpoint = torch.load(ckpt_latest, map_location=device)
    state_dict = checkpoint["generator_state_dict"] if "generator_state_dict" in checkpoint else checkpoint
    generator.load_state_dict(state_dict)
    generator.eval()

    # 3. Load Test Data
    transform = PairedTransform([
        ToTensor(),
        Normalize(
            rgb_mean=[0.5, 0.5, 0.5],
            rgb_std=[0.5, 0.5, 0.5],
            thermal_mean=[0.5],
            thermal_std=[0.5]
        )
    ])
    test_dataset = LLVIPDataset(train=False, transform=transform)

    # 4. Generate Visual Comparisons: [RGB | Real Thermal | Generated Thermal]
    print(f"Generating {args.num_samples} sample images for Epoch {epoch_num}...")
    with torch.no_grad():
        for i in range(min(args.num_samples, len(test_dataset))):
            rgb, thermal = test_dataset[i]
            rgb_tensor = rgb.unsqueeze(0).to(device)

            fake_thermal = generator(rgb_tensor)

            # Un-normalize from [-1, 1] to [0, 1] range
            rgb_vis = (rgb * 0.5 + 0.5).clamp(0, 1)
            thermal_vis = (thermal * 0.5 + 0.5).repeat(3, 1, 1).clamp(0, 1)
            fake_vis = (fake_thermal.squeeze(0).cpu() * 0.5 + 0.5).repeat(3, 1, 1).clamp(0, 1)

            # Concatenate horizontally
            comparison = torch.cat([rgb_vis, thermal_vis, fake_vis], dim=2)
            out_file = os.path.join(output_dir, f"sample_{i+1}.png")
            save_image(comparison, out_file)

    print(f"[✓] Saved sample images to: {output_dir}/")


if __name__ == "__main__":
    main()