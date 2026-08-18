import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class LLVIPDataset(Dataset):
    """Custom Dataset for the LLVIP RGB-Thermal dataset."""

    def __init__(self, train = True, transform = None):

        self.transform = transform

        # Resolve project root: src/datasets/dataset.py -> src/datasets -> src -> project root
        datasets_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(datasets_dir)
        project_root = os.path.dirname(src_dir)

        # Set path 
        self.data_path = os.path.join(project_root, "processed_data")

        if train:
            self.rgb_path = os.path.join(self.data_path, "visible/train")
            self.thermal_path = os.path.join(self.data_path, "infrared/train")
        else:
            self.rgb_path = os.path.join(self.data_path, "visible/test")
            self.thermal_path = os.path.join(self.data_path, "infrared/test")

        self.rgb_list = sorted(os.listdir(self.rgb_path))
        self.thermal_list = sorted(os.listdir(self.thermal_path))

        self.len_rgb = len(self.rgb_list)
        self.len_thermal = len(self.thermal_list)
        if self.len_rgb != self.len_thermal:
            raise ValueError(
                f"Dataset mismatch: RGB={self.len_rgb}, Thermal={self.len_thermal}"
            )
        
 
    def __len__(self):
        return self.len_rgb
        

    def __getitem__(self, index):
        
        rgb_file = os.path.join(self.rgb_path, self.rgb_list[index])
        thermal_file = os.path.join(self.thermal_path, self.thermal_list[index])
         
        # Explicit mode conversion prevents tensor shape mismatch issues
        rgb_image = Image.open(rgb_file).convert("RGB")
        thermal_image = Image.open(thermal_file).convert("L")

        if self.transform:
            rgb_image, thermal_image = self.transform(rgb_image, thermal_image)

        return rgb_image, thermal_image
    
    
    


