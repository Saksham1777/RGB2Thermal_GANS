import torch
import torch.nn as nn

class GenLoss(nn.Module):
    def __init__(self, lambda_l1=100):
        super().__init__()
        self.lambda_l1  = lambda_l1
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()
    
    def forward(self, discriminator_output, generated_thermal, real_thermal):
    
        # BCE compares discriminator logits against "real" targets
        target = torch.ones_like(discriminator_output)
        gan_loss = self.bce(discriminator_output, target)
        
        # syntax for l1 loss -> (input, target)
        l1_loss = self.l1(generated_thermal, real_thermal)
        
        return gan_loss + self.lambda_l1 * l1_loss

class DiscLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        
    # disc_real = disc patch score for real thermal image
    # disc_fake = disc patch score for genrated thermal image
    def forward(self,  disc_real, disc_fake):
        
        ones = torch.ones_like(disc_real)
        zeros = torch.zeros_like(disc_fake)

        real_loss = self.bce(disc_real, ones)
        fake_loss = self.bce(disc_fake, zeros)

        return 0.5 * (fake_loss + real_loss)