import torch
import torch.nn as nn
import torch.nn.functional as F


class GenLoss(nn.Module):
    def __init__(self, lambda_content=20.0):
        super().__init__()
        self.lambda_content = lambda_content
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()

    def ssim_loss(self, img1, img2, window_size=11):
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
        mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)

        sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size // 2) - mu1.pow(2)
        sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size // 2) - mu2.pow(2)
        sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2) - mu1 * mu2

        ssim_map = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1.pow(2) + mu2.pow(2) + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        return 1.0 - ssim_map.mean()

    def forward(self, discriminator_output, generated_thermal, real_thermal):
        target = torch.ones_like(discriminator_output)
        gan_loss = self.bce(discriminator_output, target)

        # L1 works perfectly fine on the [-1, 1] range
        l1_val = self.l1(generated_thermal, real_thermal)
        
        # THE FIX: Shift tensors from [-1, 1] to [0, 1] specifically for SSIM math
        gen_norm = generated_thermal * 0.5 + 0.5
        real_norm = real_thermal * 0.5 + 0.5
        ssim_val = self.ssim_loss(gen_norm, real_norm)

        # Balanced 50/50 content loss
        content_loss = 0.5 * l1_val + 0.5 * ssim_val

        return gan_loss + self.lambda_content * content_loss


class DiscLoss(nn.Module):
    def __init__(self, label_smoothing=0.9):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.label_smoothing = label_smoothing

    def forward(self, disc_real, disc_fake):
        # Use soft labels (0.9 instead of 1.0) to prevent discriminator overconfidence
        real_targets = torch.full_like(disc_real, self.label_smoothing)
        fake_targets = torch.zeros_like(disc_fake)

        real_loss = self.bce(disc_real, real_targets)
        fake_loss = self.bce(disc_fake, fake_targets)

        return 0.5 * (fake_loss + real_loss)