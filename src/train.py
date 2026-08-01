import torch
from models.discriminator import Discriminator
from models.generator import Generator
from training.losses import GenLoss, DiscLoss
from datasets.transform import PairedTransform, ToTensor, Normalize
from datasets.dataset import LLVIPDataset
from torch.utils.data import DataLoader
from torch.optim import Adam

def train():

        device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
        batch_size = 64 #we can decide thi
        epochs = 100 # say 100 for now- we can always change
        learning_rate = 0.0002 # place holder since we will use adam right?
        lambda_l1 = 100 # placeholder

        transform = PairedTransform ([
                        ToTensor(),
                        Normalize() #need values - added tools file for this
        ])

        # however this is wrong - i need to edit the dataset loader to take images from the resize after prprocess runs...
        train_dataset = LLVIPDataset(
                train=True,
                transform=transform
                )

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


        gen_loss = GenLoss(lambda_l1=lambda_l1)
        disc_loss = DiscLoss()

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