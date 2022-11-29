import os
import json
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .datasets import register


@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, root_path, image_size=224, box_size=256, **kwargs):
        if box_size is None:
            box_size = image_size

        self.filepaths = []
        self.label = []
        classes = sorted(os.listdir(root_path))

        if kwargs.get('split'):
            path = kwargs.get('split_file')
            if path is None:
                path = os.path.join(
                        os.path.dirname(root_path.rstrip('/')), 'split.json')
            split = json.load(open(path, 'r'))
            
            #classes = [classes[i] for i in split[kwargs['split']]]
            classes = sorted(split[kwargs['split']])

        for i, c in enumerate(classes):
            for filename in sorted(os.listdir(os.path.join(root_path, c))):
                #if kwargs.get('classes'):
                if i >= 200:
                    break
                self.filepaths.append(os.path.join(root_path, c, filename))
                self.label.append(i)
        self.n_classes = max(self.label) + 1

        norm_params = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
        normalize = transforms.Normalize(**norm_params)
        if kwargs.get('augment'):
            augment = kwargs.get('augment')
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(box_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ])

        def convert_raw(x):
            mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
            std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
            return x * std + mean
        self.convert_raw = convert_raw
        self.fine_tune = False

        self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        #self.img = [Image.open(i).convert('RGB') for i in self.filepaths]
    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, i):
        img = Image.open(self.filepaths[i]).convert('RGB')

        if self.fine_tune:
            return self.train_transform(img), self.label[i], i
        else:
        
            return self.transform(img), self.label[i], i

    def few_set(self, idx):
        self.img = {}
        for i in idx:
            self.img[i] = Image.open(self.filepaths[i]).convert('RGB')