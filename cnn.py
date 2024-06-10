#%%
import torch
from torchvision.transforms import v2
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn

#%% Loading the training set

train_transforms = v2.Compose([
    v2.Resize((64,64)),
    v2.RandomHorizontalFlip(),
    v2.RandomAffine(degrees=0,shear=0.2,scale=(0.8, 1.2)),
    v2.ToTensor(),
    v2.Normalize((0.5,), (0.5,))])

train_set = datasets.ImageFolder(
    root='dataset/training_set',
    transform=train_transforms)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=5)

#%% Loading the test set

test_transforms = v2.Compose([
    v2.Resize((64,64)),
    v2.ToTensor(),
    v2.Normalize((0.5,), (0.5,))])

test_set = datasets.ImageFolder(
    root='dataset/test_set',
    transform=test_transforms)

test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=5)

#%% Creating the CNN model

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    
#%% Creating the CNN model 
    
    
    
    
    
    
    
    
    
    
    
    
    
    