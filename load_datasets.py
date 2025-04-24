#%%
import os
import pandas as pd
from PIL import Image, ImageDraw
from torchvision import transforms
from torch.utils.data import Dataset
from medmnist import OrganAMNIST, OrganCMNIST, OrganSMNIST
from torchvision import datasets
import random 
import matplotlib.pyplot as plt 
from config import DATA_DIR 
#%%
MAIN_DATA_DIR = DATA_DIR 
#%%
def pad_zeros_at_front(num, N):
    return  str(num).zfill(N)

class RepeatChannelTransform:
    def __call__(self, img):
        return img.repeat(3, 1, 1)
    
class DatasetWithImageID(Dataset):
    def __init__(self, dataset, dataset_name):
        self.dataset = dataset
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image_id = f'{self.dataset_name}_{pad_zeros_at_front(idx, 6)}'
        return image, label, image_id
#%%
def get_mnist_dataset(split='train', img_size=None, convert_to_three_channels=False):
    if img_size is not None:
        if convert_to_three_channels == True:
            transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Resize(img_size),
                transforms.Normalize((0.1307,), (0.3081,)),
                RepeatChannelTransform(),
                ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Resize(img_size),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(), 
        ])
    
    if split == 'train':
        dataset = datasets.MNIST(root=MAIN_DATA_DIR, train=True, download=True, transform=transform)
    elif split == 'test':
        dataset = datasets.MNIST(root=MAIN_DATA_DIR, train=False, download=True, transform=transform)
    else:
        print(f'Invalid split value: {split}')
    return dataset 


#%%
def get_organcmnist_dataset(split='train', img_size=64):
    dataset = OrganCMNIST(split=split, root=MAIN_DATA_DIR, download=True, size=img_size)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure the image is grayscale
        transforms.ToTensor(),  # Convert PIL image to torch tensor
        transforms.Resize((img_size, img_size)), 
    ])
    
    dataset.transform = transform
    
    return dataset