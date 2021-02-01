'''
proxy-synthesis
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
import os
import glob
import random
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.datasets as datasets
from torch.utils.data.sampler import Sampler

class define_dataset(data.Dataset):
    def __init__(self, dir_path, n_instance=1, transforms=None):
        
        self.dir_path = dir_path
        self.n_instance = n_instance
        self.transforms = transforms

        label_folders = sorted(glob.glob(os.path.join(dir_path, '*')))
        
        self.label_idx_dict = {}

        self.image_list = []
        
        image_idx = 0
        for label_idx, label_folder in enumerate(label_folders):
            idx_list = []
            for image_path in glob.glob(os.path.join(label_folder, '*')):
                idx_list.append(image_idx)
                self.image_list.append([image_path, label_idx])
                image_idx += 1

            self.label_idx_dict[label_idx] = idx_list

    def __getitem__(self, index):
        image_path, label_idx = self.image_list[index]

        image = Image.open(image_path).convert('RGB')
        
        if self.transforms is not None:
            image = self.transforms(image)

        return image, label_idx

    def __len__(self):
        return len(self.image_list)

class sampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.label_idx_dict = dataset.label_idx_dict
        self.batch_size = batch_size
        self.n_instance = dataset.n_instance
        self.len_dataset = len(dataset)
        self.total_iter = len(dataset) // batch_size
        self.key_list = list(self.label_idx_dict.keys())
        
    def __iter__(self):
        for _ in range(self.total_iter):
            n_labels = self.batch_size // self.n_instance # the number of selected classes in mini-batch
            
            if n_labels < len(self.key_list):
                selected_labels = random.sample(self.key_list, n_labels)
            else:
                selected_labels = np.random.choice(self.key_list, n_labels, replace=True)

            selected_idx = [idx for label in selected_labels for idx in random.sample(self.label_idx_dict[label], self.n_instance)]
            
            for idx in selected_idx[:self.batch_size]:
                yield idx

    
    def __len__(self):
        return self.len_dataset

def call_train_loader(traindir, args, transforms):
    if args.n_instance > 1:
        train_dataset = define_dataset(traindir, args.n_instance, transforms)
        train_sampler = sampler(train_dataset, args.batch_size)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   sampler=train_sampler,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.workers,
                                                   pin_memory=True)
    else:
        train_dataset = datasets.ImageFolder(traindir, transforms)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.workers,
                                                   pin_memory=True)

    return train_loader


def get_class_dict(dataset):
    imgs = dataset.imgs
    class_dict = {} # key: class_idx | value: the number of images of class_idx

    for img_info in imgs:
        cls_idx = img_info[1]
        if cls_idx in class_dict:
            class_dict[cls_idx] += 1
        else:
            class_dict[cls_idx] = 1
    
    return class_dict, max(class_dict.values())
