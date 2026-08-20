import torch
import torch.nn as nn


class GenLoss(nn.Module):
    def __init__(self, lambda_content=100.0):
        super().__init__()
        self.lambda_content = lambda_content
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()

    def forward(self, discriminator_output, generated_thermal, real_thermal):
        # Adversarial GAN Loss (Fool Discriminator)
        target = torch.ones_like(discriminator_output)
        gan_loss = self.bce(discriminator_output, target)

        # Standard L1 Content Loss across [-1, 1] range
        l1_val = self.l1(generated_thermal, real_thermal)

        return gan_loss + self.lambda_content * l1_val


class DiscLoss(nn.Module):
    def __init__(self, label_smoothing=0.9):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.label_smoothing = label_smoothing

    def forward(self, disc_real, disc_fake):
        real_targets = torch.full_like(disc_real, self.label_smoothing)
        fake_targets = torch.zeros_like(disc_fake)

        real_loss = self.bce(disc_real, real_targets)
        fake_loss = self.bce(disc_fake, fake_targets)

        return 0.5 * (fake_loss + real_loss)