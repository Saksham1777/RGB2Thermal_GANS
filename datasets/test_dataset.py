import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from dataset import LLVIPDataset

data = LLVIPDataset(train = True)

print(__file__)
print(len(data))
print(type(data[0]))
rgb, thermal = data[0]
print(type(rgb))
print(type(thermal))