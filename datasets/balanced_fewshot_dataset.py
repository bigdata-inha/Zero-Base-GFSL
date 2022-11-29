import os
import pickle
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

from .datasets import register

@register('balanced-few')
class BalancedFew(Dataset):

    def __init__(self, data, label, image_size=84, **kwargs):
   
        
        self.data = data
        self.label = label
        self.n_classes = max(self.label) + 1

        norm_params = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
        normalize = transforms.Normalize(**norm_params)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        def convert_raw(x):
            mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
            std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
            return x * std + mean
        self.convert_raw = convert_raw



        self.fine_tune = True
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):

        return self.transform(self.data[i]), self.label[i]

