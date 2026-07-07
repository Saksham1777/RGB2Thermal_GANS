import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class LLVIPDataset(Dataset):
    """Custom Dataset for the LLVIP RGB-Thermal dataset."""

    def __init__(self, train = True):
        
        dataset_path = os.path.dirname(__file__)
        self.base_path = os.path.dirname(dataset_path)
        self.data_path = os.path.join(self.base_path, "data")
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
         
        rgb_image = Image.open(rgb_file)
        thermal_image = Image.open(thermal_file)

        return rgb_image, thermal_image
    
    
    


