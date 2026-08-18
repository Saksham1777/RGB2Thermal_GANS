import torch
import os
from models.discriminator import Discriminator
from models.generator import Generator
from training.losses import GenLoss, DiscLoss
from datasets.transform import PairedTransform, ToTensor, Normalize
from datasets.dataset import LLVIPDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

def train():

        writer = SummaryWriter("runs/rgb2thermal")

        device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
        batch_size = 64 #we can decide thi
        epochs = 100 # say 100 for now- we can always change
        learning_rate = 0.0002 # place holder since we will use adam right?
        lambda_l1 = 100 # placeholder

        os.makedirs("saved_models", exist_ok=True)

        transform = PairedTransform ([
                        ToTensor(),
                        Normalize() #need values - added tools file for this
        ])

        # however this is wrong - i need to edit the dataset loader to take images from the resize after prprocess runs...
        train_dataset = LLVIPDataset(
                train=True,
                transform=transform
                )
        # batch of 64 of (rgb_batch, thermal_batch)

        train_dataloader = DataLoader(
                train_dataset, 
                batch_size = batch_size, 
                shuffle = True, 
                num_workers = 0
        )

        test_dataset = LLVIPDataset(
                train=False,
                transform=transform
                )

        test_dataloader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
        )


        # Build network of gen, disc  
        generator = Generator().to(device)
        discriminator = Discriminator().to(device)


        gen_loss_fn = GenLoss(lambda_l1=lambda_l1).to(device)
        disc_loss_fn = DiscLoss().to(device)

        optimizer_G = Adam(
                generator.parameters(),
                lr=learning_rate,
                betas=(0.5, 0.999)
        )

        optimizer_D = Adam(
                discriminator.parameters(),
                lr=learning_rate,
                betas=(0.5, 0.999)
        )

        generator.train()
        discriminator.train()  

        

        all_d_losses = []
        all_g_losses = []

        for epoch in range(epochs):

                epoch_d_losses = []
                epoch_g_losses = []

                for batch_idx, (rgb, thermal) in enumerate(train_dataloader):

                        # keep image and model on same device
                        rgb = rgb.to(device)
                        thermal = thermal.to(device)

                        fake_thermal = generator(rgb)

                        score_real = discriminator(rgb, thermal)

                        # Evaluate the discriminator on a generated RGB-Thermal pair.
                        # detach() prevents gradients from flowing back into the generator
                        # while we are training only the discriminator.
                        score_fake = discriminator(rgb, fake_thermal.detach())
                        
                        # Compute how well the discriminator distinguished
                        # real pairs from generated pairs.
                        disc_loss = disc_loss_fn(score_real, score_fake) 

                        # Remove gradients accumulated from the previous iteration.
                        optimizer_D.zero_grad()

                        # Compute gradients for the discriminator parameters.
                        disc_loss.backward()

                        # Update only the discriminator weights using Adam.
                        optimizer_D.step()

                        # re score using on new discriminator (new weights),
                        # not detaching so that we can train genrator
                        score_fake = discriminator(rgb, fake_thermal)

                        gen_loss = gen_loss_fn(score_fake, fake_thermal, thermal)

                        optimizer_G.zero_grad()

                        gen_loss.backward()

                        optimizer_G.step()

                        epoch_d_losses.append(disc_loss.item())
                        epoch_g_losses.append(gen_loss.item())

                        if batch_idx % 10 == 0:
                                print(
                                        f"Epoch [{epoch+1}/{epochs}] "
                                        f"Batch [{batch_idx}/{len(train_dataloader)}] "
                                        f"D Loss: {disc_loss.item():.4f} "
                                        f"G Loss: {gen_loss.item():.4f}"
                                )  

                avg_d_loss = sum(epoch_d_losses) / len(epoch_d_losses)
                avg_g_loss = sum(epoch_g_losses) / len(epoch_g_losses)

                all_d_losses.append(avg_d_loss)
                all_g_losses.append(avg_g_loss)

                writer.add_scalar("Loss/Discriminator", avg_d_loss, epoch + 1)
                writer.add_scalar("Loss/Generator", avg_g_loss, epoch + 1)                  

                if (epoch + 1) % 25 == 0 or (epoch + 1) == epochs:
                        torch.save(
                                generator.state_dict(),
                                f"saved_models/generator_epoch_{epoch+1}.pth"
                        )

                        torch.save(
                                discriminator.state_dict(),
                                f"saved_models/discriminator_epoch_{epoch+1}.pth"
                        )

        writer.close()                

        
if __name__ == "__main__":
    train()