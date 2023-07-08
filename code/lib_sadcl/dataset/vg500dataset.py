import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
from pdb import set_trace as stop
import json
from PIL import Image
import time

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def image_loader(path,transform):
    try:
        image = Image.open(path)
    except FileNotFoundError: # weird issues with loading images on our servers
        # print('FILE NOT FOUND')
        time.sleep(10)
        image = Image.open(path)

    image = image.convert('RGB')

    if transform is not None:
        image = transform(image)

    return image

class VGDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, img_list, image_transform, label_path):
        with open(img_list, 'r') as f:
            self.img_names = f.readlines()
        with open(label_path, 'r') as f:
            self.labels = json.load(f) 
        
        self.image_transform = image_transform
        self.img_dir = img_dir
        self.num_labels= 500

    
    def __getitem__(self, index):
        name = self.img_names[index][:-1]
        img_path = os.path.join(self.img_dir, name)
        image = image_loader(img_path, self.image_transform)

        label = np.zeros(self.num_labels).astype(np.float32)
        label[self.labels[name]] = 1.0
        label = torch.Tensor(label)


        return image, label

    def __len__(self):
        return len(self.img_names)