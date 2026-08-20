import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision.utils import make_grid

# Adjust these imports to match your project module names/paths
from dataset import RGB2ThermalDataset  # Your PyTorch Dataset class
from models import Generator, Discriminator  # Your Generator and Discriminator models
from training.losses import GenLoss, DiscLoss


def train():
    # -------------------------------------------------------------------------
    # 1. Device Setup & Hyperparameters
    # -------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_epochs = 100
    batch_size = 8
    lr = 2e-4
    beta1 = 0.5
    beta2 = 0.999
    lambda_content = 100.0  # Standard Pix2Pix L1 loss weight

    checkpoint_dir = "saved_models"
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_latest.pth")

    # -------------------------------------------------------------------------
    # 2. Data & TensorBoard Initialization
    # -------------------------------------------------------------------------
    writer = SummaryWriter("runs/rgb2thermal")

    # Update data_dir to match your actual dataset location
    train_dataset = RGB2ThermalDataset(data_dir="data/train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
    )

    # -------------------------------------------------------------------------
    # 3. Model, Loss, & Optimizer Setup
    # -------------------------------------------------------------------------
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    gen_loss_fn = GenLoss(lambda_content=lambda_content).to(device)
    disc_loss_fn = DiscLoss(label_smoothing=0.9).to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    start_epoch = 1

    # -------------------------------------------------------------------------
    # 4. Checkpoint Resuming Logic
    # -------------------------------------------------------------------------
    if os.path.exists(latest_checkpoint_path):
        print(f"Loading checkpoint from {latest_checkpoint_path}...")
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)

        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_G.load_state_dict(checkpoint["optimizer_G"])
        optimizer_D.load_state_dict(checkpoint["optimizer_D"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Successfully resumed! Continuing from epoch {start_epoch}.")

    # -------------------------------------------------------------------------
    # 5. Main Training Loop
    # -------------------------------------------------------------------------
    for epoch in range(start_epoch, num_epochs + 1):
        generator.train()
        discriminator.train()

        running_d_loss = 0.0
        running_g_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")

        for batch_idx, (rgb_img, real_thermal) in enumerate(pbar):
            rgb_img = rgb_img.to(device)
            real_thermal = real_thermal.to(device)

            # -----------------------------------------------------------------
            # Step A: Update Discriminator
            # -----------------------------------------------------------------
            optimizer_D.zero_grad()

            fake_thermal = generator(rgb_img)

            # Evaluate real pair and fake pair
            disc_real = discriminator(rgb_img, real_thermal)
            disc_fake = discriminator(rgb_img, fake_thermal.detach())

            d_loss = disc_loss_fn(disc_real, disc_fake)
            d_loss.backward()
            optimizer_D.step()

            # -----------------------------------------------------------------
            # Step B: Update Generator
            # -----------------------------------------------------------------
            optimizer_G.zero_grad()

            disc_fake_for_g = discriminator(rgb_img, fake_thermal)

            g_loss = gen_loss_fn(disc_fake_for_g, fake_thermal, real_thermal)
            g_loss.backward()

            # GRADIENT CLIPPING: Prevents exploding gradient spikes from ruining weights
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)

            optimizer_G.step()

            # Track metrics
            running_d_loss += d_loss.item()
            running_g_loss += g_loss.item()

            pbar.set_postfix(
                D_Loss=f"{d_loss.item():.4f}", G_Loss=f"{g_loss.item():.4f}"
            )

        # Average losses for TensorBoard logging
        epoch_d_loss = running_d_loss / len(train_loader)
        epoch_g_loss = running_g_loss / len(train_loader)

        writer.add_scalar("Loss/Discriminator", epoch_d_loss, epoch)
        writer.add_scalar("Loss/Generator", epoch_g_loss, epoch)

        # -------------------------------------------------------------------------
        # 6. Visual Logging & Checkpoint Saving
        # -------------------------------------------------------------------------
        if epoch % 5 == 0 or epoch == num_epochs:
            generator.eval()
            with torch.no_grad():
                # Fetch a small sample batch for evaluation display
                sample_rgb, sample_real = next(iter(train_loader))
                sample_rgb = sample_rgb[:4].to(device)
                sample_real = sample_real[:4].to(device)
                sample_fake = generator(sample_rgb)

                # Rescale [-1, 1] -> [0, 1] for visual display
                sample_rgb = sample_rgb * 0.5 + 0.5
                sample_real = sample_real * 0.5 + 0.5
                sample_fake = sample_fake * 0.5 + 0.5

                # Channel match 1-channel thermal to 3-channel for grid display
                if sample_real.shape[1] == 1:
                    sample_real = sample_real.repeat(1, 3, 1, 1)
                    sample_fake = sample_fake.repeat(1, 3, 1, 1)

                comparison_grid = make_grid(
                    torch.cat([sample_rgb, sample_real, sample_fake], dim=0),
                    nrow=4,
                )
                writer.add_image(
                    "Visual_Comparison (RGB | Real Thermal | Fake Thermal)",
                    comparison_grid,
                    epoch,
                )

            # Save milestone checkpoint
            checkpoint_path = os.path.join(
                checkpoint_dir, f"checkpoint_epoch{epoch}.pth"
            )
            state = {
                "epoch": epoch,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optimizer_G": optimizer_G.state_dict(),
                "optimizer_D": optimizer_D.state_dict(),
            }
            torch.save(state, checkpoint_path)

        # Always update checkpoint_latest.pth
        latest_state = {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_G": optimizer_G.state_dict(),
            "optimizer_D": optimizer_D.state_dict(),
        }
        torch.save(latest_state, latest_checkpoint_path)

    writer.close()
    print("Training process finished!")


if __name__ == "__main__":
    train()