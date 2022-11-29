import os
import pickle
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

from .datasets import register

@register('mini-imagenet')
class MiniImageNet(Dataset):

    def __init__(self, root_path, split='train', **kwargs):
        split_tag = split
        if split == 'train':
            split_tag = 'train_phase_train'
        split_file = 'miniImageNet_category_split_{}.pickle'.format(split_tag)
        with open(os.path.join(root_path, split_file), 'rb') as f:
            pack = pickle.load(f, encoding='latin1')
        data = pack['data']
        label = pack['labels']

        image_size = 84
        data = [Image.fromarray(x) for x in data]

        min_label = min(label)
        label = [x - min_label for x in label]
        
        self.data = data
        self.label = label
        self.n_classes = max(self.label) + 1

        norm_params = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
        normalize = transforms.Normalize(**norm_params)
        self.default_transform = transforms.Compose([
            transforms.Resize(92),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            normalize,  
        ])
        augment = kwargs.get('augment')
        if augment == 'resize':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.transform = self.default_transform

        def convert_raw(x):
            mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
            std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
            return x * std + mean
        self.convert_raw = convert_raw

        # train이 crop이면 base 상승
        self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,  
                ])  

        self.fine_tune = False
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if self.fine_tune:
            return self.train_transform(self.data[i]), self.label[i], i
        else:
            return self.transform(self.data[i]), self.label[i], i

