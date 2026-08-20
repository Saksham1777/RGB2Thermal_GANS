import os

# Prevent PyTorch from fragmenting 4GB VRAM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.discriminator import Discriminator
from models.generator import Generator
from training.losses import GenLoss, DiscLoss
from datasets.transform import PairedTransform, ToTensor, Normalize
from datasets.dataset import LLVIPDataset


def train():
    writer = SummaryWriter("runs/rgb2thermal")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optimize cuDNN algorithms for fixed resolution inputs
    torch.backends.cudnn.benchmark = True

    batch_size = 8          # Fits safely inside 4GB VRAM
    epochs = 100
    
    # TTUR: Generator learns faster than Discriminator
    g_learning_rate = 0.0002
    d_learning_rate = 0.0001
    
    lambda_content = 50.0   # Balanced content weight (L1 + SSIM)

    os.makedirs("saved_models", exist_ok=True)

    # Dynamic file path resolution relative to train.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "tools", "normalization_stats.json")

    with open(json_path, "r") as f:
        stats = json.load(f)

    transform = PairedTransform([
        ToTensor(),
        Normalize(
            rgb_mean=[0.5, 0.5, 0.5],
            rgb_std=[0.5, 0.5, 0.5],
            thermal_mean=[0.5],
            thermal_std=[0.5]
        )
    ])

    train_dataset = LLVIPDataset(train=True, transform=transform)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False
    )

    # Build models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    gen_loss_fn = GenLoss(lambda_content=lambda_content, ssim_weight=0.5).to(device)
    disc_loss_fn = DiscLoss(label_smoothing=0.9).to(device)

    optimizer_G = Adam(generator.parameters(), lr=g_learning_rate, betas=(0.5, 0.999))
    optimizer_D = Adam(discriminator.parameters(), lr=d_learning_rate, betas=(0.5, 0.999))

    scaler_G = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    scaler_D = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    start_epoch = 0
    best_g_loss = float("inf")
    resume_path = "saved_models/checkpoint_latest.pth"

    # Load previous training state if checkpoint exists
    if os.path.exists(resume_path):
        print(f"Found existing checkpoint at '{resume_path}'. Loading state...")
        checkpoint = torch.load(resume_path, map_location=device)

        generator.load_state_dict(checkpoint["generator_state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        
        # Load optimizer states, updating LR in place
        optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
        for param_group in optimizer_G.param_groups:
            param_group['lr'] = g_learning_rate
        for param_group in optimizer_D.param_groups:
            param_group['lr'] = d_learning_rate

        scaler_G.load_state_dict(checkpoint["scaler_G_state_dict"])
        scaler_D.load_state_dict(checkpoint["scaler_D_state_dict"])

        start_epoch = checkpoint["epoch"]
        best_g_loss = checkpoint.get("best_g_loss", float("inf"))
        print(f"Successfully resumed! Continuing from epoch {start_epoch + 1}.")
    else:
        print("No previous checkpoint found. Starting fresh training run.")

    generator.train()
    discriminator.train()

    try:
        for epoch in range(start_epoch, epochs):
            epoch_d_losses = []
            epoch_g_losses = []

            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

            for batch_idx, (rgb, thermal) in enumerate(pbar):
                rgb = rgb.to(device, non_blocking=True)
                thermal = thermal.to(device, non_blocking=True)

                # --- Train Discriminator ---
                optimizer_D.zero_grad()

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    fake_thermal = generator(rgb)
                    score_real = discriminator(rgb, thermal)
                    score_fake = discriminator(rgb, fake_thermal.detach())
                    disc_loss = disc_loss_fn(score_real, score_fake)

                scaler_D.scale(disc_loss).backward()
                scaler_D.step(optimizer_D)
                scaler_D.update()

                # --- Train Generator ---
                optimizer_G.zero_grad()

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    score_fake = discriminator(rgb, fake_thermal)
                    gen_loss = gen_loss_fn(score_fake, fake_thermal, thermal)

                scaler_G.scale(gen_loss).backward()
                scaler_G.step(optimizer_G)
                scaler_G.update()

                # Divergence Guard
                if torch.isnan(disc_loss) or torch.isnan(gen_loss):
                    print(f"\n[DIVERGENCE ALERT] NaN loss detected at epoch {epoch+1}, batch {batch_idx}! Stopping training.")
                    return

                epoch_d_losses.append(disc_loss.item())
                epoch_g_losses.append(gen_loss.item())

                pbar.set_postfix({
                    "D_Loss": f"{disc_loss.item():.4f}",
                    "G_Loss": f"{gen_loss.item():.4f}"
                })

            avg_d_loss = sum(epoch_d_losses) / len(epoch_d_losses)
            avg_g_loss = sum(epoch_g_losses) / len(epoch_g_losses)

            writer.add_scalar("Loss/Discriminator", avg_d_loss, epoch + 1)
            writer.add_scalar("Loss/Generator", avg_g_loss, epoch + 1)

            # Save full state checkpoint
            checkpoint_data = {
                "epoch": epoch + 1,
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "optimizer_G_state_dict": optimizer_G.state_dict(),
                "optimizer_D_state_dict": optimizer_D.state_dict(),
                "scaler_G_state_dict": scaler_G.state_dict(),
                "scaler_D_state_dict": scaler_D.state_dict(),
                "best_g_loss": best_g_loss,
            }
            torch.save(checkpoint_data, resume_path)

            if avg_g_loss < best_g_loss:
                best_g_loss = avg_g_loss
                checkpoint_data["best_g_loss"] = best_g_loss
                torch.save(generator.state_dict(), "saved_models/generator_best.pth")

            # TensorBoard logging
            if (epoch + 1) % 5 == 0:
                with torch.no_grad():
                    rgb_vis = rgb[:4].detach().cpu() * 0.5 + 0.5
                    thermal_vis = thermal[:4].detach().cpu() * 0.5 + 0.5
                    fake_vis = fake_thermal[:4].detach().cpu() * 0.5 + 0.5

                    writer.add_images("RGB_Input", torch.clamp(rgb_vis, 0, 1), epoch + 1)
                    writer.add_images("Thermal_Real", torch.clamp(thermal_vis, 0, 1), epoch + 1)
                    writer.add_images("Thermal_Generated", torch.clamp(fake_vis, 0, 1), epoch + 1)

            if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
                torch.save(generator.state_dict(), f"saved_models/generator_epoch_{epoch+1}.pth")
                torch.save(discriminator.state_dict(), f"saved_models/discriminator_epoch_{epoch+1}.pth")

    except KeyboardInterrupt:
        print("\n[Manual Stop] Training interrupted. Saving emergency checkpoint...")
        torch.save(generator.state_dict(), "saved_models/generator_interrupted.pth")
        torch.save(discriminator.state_dict(), "saved_models/discriminator_interrupted.pth")
        print("Saved 'generator_interrupted.pth' successfully!")

    finally:
        writer.close()


if __name__ == "__main__":
    train()