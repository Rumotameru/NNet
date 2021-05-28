import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision import datasets, models
import torchvision.transforms as transforms
import os
import torch.nn as nn
import matplotlib.image as img
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


means = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

trans = transforms.Compose([transforms.ToPILImage(),
                            transforms.ToTensor(),
                            transforms.Normalize(means, std),
                            transforms.Resize([150, 150]),
                            transforms.RandomRotation(90)])
pr_trans = transforms.Compose([transforms.ToPILImage(),
                            transforms.ToTensor(),
                            transforms.Normalize(means, std),
                            transforms.Resize([150, 150])])

def train_info_load(path):
    images = []
    labels = []
    cr_dir = os.getcwd()
    for dirname, _, filenames in os.walk(os.path.join(cr_dir, path)):
        for filename in filenames:
            images.append(filename)
            if filename.find('all') != -1:
                labels.append(1)
            elif filename.find('hem') != -1:
                labels.append(0)

    info = pd.DataFrame({'images': images, 'labels': labels})
    return info


def test_info_load(path, filename):
    cr_dir = os.getcwd()
    info = pd.read_csv(os.path.join(cr_dir, path, filename))
    info = info.drop('Patient_ID', axis=1)
    info.columns = ['images', 'labels']
    return info


class AllDataset(Dataset):
    def __init__(self, data, path, transform=None):
        super().__init__()
        self.data = data.values
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name, label = self.data[index]
        img_path = os.path.join(self.path, img_name)
        image = img.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def dataset_prepare(train_info, test_info, train_path, test_path):
    train, valid = train_test_split(train_info, stratify=train_info.labels, test_size=0.33)
    train_data = AllDataset(train, train_path, trans)
    valid_data = AllDataset(valid, train_path, trans)
    test_data = AllDataset(test_info, test_path, trans)
    return train_data, valid_data, test_data


class Predicted(Dataset):
    def __init__(self, path, transform=None):
        super().__init__()
        self.path = path
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, index):
        img_path= self.path
        image = img.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image


def prediction_data(image):
    return DataLoader(dataset=Predicted(image, pr_trans), batch_size=1)
