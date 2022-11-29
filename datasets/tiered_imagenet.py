import os
import pickle
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .datasets import register


@register('tiered-imagenet')
class TieredImageNet(Dataset):

    def __init__(self, root_path, split='train', mini=False, **kwargs):
        split_tag = split
        data = np.load(os.path.join(
                root_path, '{}_images.npz'.format(split_tag)),
                allow_pickle=True)['images']
        data = data[:, :, :, ::-1]
        with open(os.path.join(
                root_path, '{}_labels.pkl'.format(split_tag)), 'rb') as f:
            label = pickle.load(f)['labels']
        '''if 'train' in split_tag:
            b_data = np.load(os.path.join(
                root_path, '{}_images.npz'.format(split_tag)),
                allow_pickle=True)['images']
            with open(os.path.join(
                root_path, '{}_labels.pkl'.format(split_tag)), 'rb') as f:
            b_label = pickle.load(f)['label_specific']
            data = data + b_data
            label = label + b_label'''
            
        image_size = 84
        data = [Image.fromarray(x) for x in data]

        min_label = min(label)
        label = [x - min_label for x in label]

        if mini:
            data_ = []
            label_ = []
            np.random.seed(0)
            c = np.random.choice(max(label) + 1, 64, replace=False).tolist()
            n = len(data)
            cnt = {x: 0 for x in c}
            ind = {x: i for i, x in enumerate(c)}
            for i in range(n):
                y = int(label[i])
                if y in c and cnt[y] < 600:
                    data_.append(data[i])
                    label_.append(ind[y])
                    cnt[y] += 1
            data = data_
            label = label_

        self.data = data
        self.label = label
        self.n_classes = max(self.label) + 1

        norm_params = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
        normalize = transforms.Normalize(**norm_params)
        self.default_transform = transforms.Compose([
            transforms.Resize(92),
            transforms.CenterCrop(image_size),
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


