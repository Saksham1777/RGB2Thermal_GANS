from PIL import Image
import torch
from torchvision import transforms


class PairedTransform:
    """Apply a sequence of paired transforms to RGB and thermal images."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, rgb, thermal):
        for transform in self.transforms:
            rgb, thermal = transform(rgb, thermal)
        return rgb, thermal


class Resize:
    """Resize both RGB and thermal images."""

    def __init__(self, size):
        self.size = size

    def __call__(self, rgb: Image.Image, thermal: Image.Image):
        rgb = rgb.resize(self.size)
        thermal = thermal.resize(self.size)
        return rgb, thermal


class ToTensor:
    """Convert both images to PyTorch tensors."""

    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, rgb: Image.Image, thermal: Image.Image):
        rgb = self.to_tensor(rgb)
        thermal = self.to_tensor(thermal)
        return rgb, thermal


class Normalize:
    """Normalize RGB and thermal tensors independently."""

    def __init__(self, rgb_mean, rgb_std, thermal_mean, thermal_std):
        self.rgb_normalize = transforms.Normalize(
            mean = rgb_mean,
            std = rgb_std,
        )
        self.thermal_normalize = transforms.Normalize(
            mean = thermal_mean,
            std = thermal_std,
        )

    def __call__(self, rgb: torch.Tensor, thermal: torch.Tensor):
        rgb = self.rgb_normalize(rgb)
        thermal = self.thermal_normalize(thermal)
        return rgb, thermal