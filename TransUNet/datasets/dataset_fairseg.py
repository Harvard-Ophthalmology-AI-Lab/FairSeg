import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
# from scipy.ndimage.interpolation import zoom
from scipy.ndimage import zoom

from torch.utils.data import Dataset
from einops import repeat
from icecream import ic

# remove this after debugging
import argparse
from torchvision import transforms
import argparse
import logging
import os
import random
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
from icecream import ic
from PIL import Image

#########


hashmap = {-1:1, -2:2, 0:0}

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

def crop_center(img, cropx, cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

class RandomGenerator(object):
    def __init__(self, output_size, center_crop_size, use_normalize=False):
        self.output_size = output_size
        self.a_min, self.a_max = 0, 255
        self.use_normalize = use_normalize
        self.center_crop_size = center_crop_size

    def __call__(self, sample):
        image, label, attr_label, pid = sample['image'], sample['label'], sample['attr_label'], sample['pid'] 


        image = np.clip(image, self.a_min, self.a_max)
        if self.use_normalize:
            assert self.a_min != self.a_max
            image = (image - self.a_min) / (self.a_max - self.a_min)     

        ## convert label to training format
        for k in sorted(hashmap.keys()):
            label[label == k] = hashmap[k]
        
        image = crop_center(image, self.center_crop_size, self.center_crop_size)
        label = crop_center(label, self.center_crop_size, self.center_crop_size)
        
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image = repeat(image, 'c h w -> (repeat c) h w', repeat=3)
        label = torch.from_numpy(label.astype(np.float32))
        attr_label = torch.tensor(attr_label).long()
        sample = {'image': image, 'label': label.long(), 'attr_label': attr_label, 'pid': pid}
        
        return sample



class TestGenerator(object):
    def __init__(self, output_size, low_res, center_crop_size, use_normalize=False):
        self.output_size = output_size
        self.low_res = low_res
        self.a_min, self.a_max = 0, 255
        self.use_normalize = use_normalize
        self.center_crop_size = center_crop_size

    def __call__(self, sample):
        image, label, attr_label, pid = sample['image'], sample['label'], sample['attr_label'], sample['pid'] 

        image = np.clip(image, self.a_min, self.a_max)
        if self.use_normalize:
            assert self.a_min != self.a_max
            image = (image - self.a_min) / (self.a_max - self.a_min)     

        ## convert label to training format
        for k in sorted(hashmap.keys()):
            label[label == k] = hashmap[k]
        
        image = crop_center(image, self.center_crop_size, self.center_crop_size)
        label = crop_center(label, self.center_crop_size, self.center_crop_size)
      
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label_h, label_w = label.shape
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image = repeat(image, 'c h w -> (repeat c) h w', repeat=3)
        label = torch.from_numpy(label.astype(np.float32))
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32))
        attr_label = torch.tensor(attr_label).long()
        sample = {'image': image, 'label': label.long(), 'low_res_label': low_res_label.long(), 'attr_label': attr_label, 'pid': pid}
        return sample

attr_to_race = {2: 0, 3: 1, 7:2}
attr_to_language = {0: 0, 1: 1, 2:2, -1:-1}

class FairSeg_dataset(Dataset):
    def __init__(self, base_dir, split, args, balanced=False, bal_attr='race', \
                 resolution=224, transform=None, attr_label='race', img_type='fundus_slo'):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.data_dir = base_dir
        self.args = args
        self.img_type = img_type
        self.needBalance = balanced
        
        list_dir = args.list_dir
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        
        self.attr_label = attr_label
        
        self.bal_attr = bal_attr
        self.balance_factor = 1.
        self.label_samples = dict()
        self.class_samples_num = None
        self.balanced_max = 0
        self.per_attr_samples = dict()
        
        # all_files = self.find_all_files(self.data_dir, suffix='npz')
        
        self.resolution = resolution
        if self.attr_label == 'race' or self.attr_label == 'language': 
            self.sens_classes = 3
        else:
            self.sens_classes = 2

        if self.split == 'train' and self.needBalance:
            # all_files = all_files[:8000]
            self.data_files = self.bal_samples_based_attr(self.sample_list)
        else: # testing set
            # self.data_files = all_files[8000:]
            self.data_files = self.sample_list

    def __len__(self):
        return len(self.data_files)
    
   
    def find_all_files(self, folder, suffix='npz'):
        files = [f for f in os.listdir(folder) \
                 if os.path.isfile(os.path.join(folder, f)) and \
                    os.path.join(folder, f).endswith(suffix)]
        return files

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.data_files[idx].strip('\n'))
        
        data = np.load(data_path, allow_pickle=True)
        
        image, label = data[self.img_type], data['disc_cup']
        
        attr_label = data[self.attr_label].item()
       
        if self.attr_label == "age":
            attr_label=attr_label/365
            if attr_label < 60:
                attr_label = 0
            else:
                attr_label = 1
        elif self.attr_label == "maritalstatus":
            if attr_label != 0 and attr_label != -1: 
                attr_label = 1
        elif self.attr_label == 'race':
            attr_label = attr_to_race[attr_label]
        elif self.attr_label == 'language':
            attr_label = attr_to_language[attr_label]
        
        pid = data['pid'].item()
       
        sample = {'image': image, 'label': label, 'attr_label': attr_label, 'pid': pid}
        if self.transform:
            sample = self.transform(sample)
        
        return sample



def find_all_files(folder, suffix='npz'):
    files = [f for f in os.listdir(folder) \
                if os.path.isfile(os.path.join(folder, f)) and \
                os.path.join(folder, f).endswith(suffix)]
    return files
