import torch
import torch.nn as nn
import torch.nn.functional as F


class GenLoss(nn.Module):
    def __init__(self, lambda_content=50.0, ssim_weight=0.5, grad_weight=0.2):
        super().__init__()
        self.lambda_content = lambda_content
        self.ssim_weight = ssim_weight
        self.grad_weight = grad_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()

    def ssim_loss(self, img1, img2, window_size=11):
        """Calculates Structural Similarity (SSIM) loss to enforce edge and structural clarity."""
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

    def gradient_loss(self, gen, real):
        """Calculates pixel intensity differences along horizontal and vertical axes (edges)."""
        dh_gen = torch.abs(gen[:, :, :, 1:] - gen[:, :, :, :-1])
        dh_real = torch.abs(real[:, :, :, 1:] - real[:, :, :, :-1])
        dv_gen = torch.abs(gen[:, :, 1:, :] - gen[:, :, :-1, :])
        dv_real = torch.abs(real[:, :, 1:, :] - real[:, :, :-1, :])
        return torch.mean(torch.abs(dh_gen - dh_real)) + torch.mean(torch.abs(dv_gen - dv_real))

    def forward(self, discriminator_output, generated_thermal, real_thermal):
        # BCE compares discriminator logits against "real" targets
        target = torch.ones_like(discriminator_output)
        gan_loss = self.bce(discriminator_output, target)

        l1_val = self.l1(generated_thermal, real_thermal)
        ssim_val = self.ssim_loss(generated_thermal, real_thermal)
        grad_val = self.gradient_loss(generated_thermal, real_thermal)

        # Correctly normalized weights: 30% L1, 50% SSIM, 20% Gradient Difference Loss (Sum = 1.0)
        l1_weight = 1.0 - (self.ssim_weight + self.grad_weight)
        content_loss = l1_weight * l1_val + self.ssim_weight * ssim_val + self.grad_weight * grad_val

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