#Train and test CIFAR with pytorch’s inbuilt batch normalization and 14 layers
# python3 train_cifar.py --normalization torch_bn --n 2 --data_dir  ../data/cifar-10-batches-py --output_file trained_models_cifar/part_1.1.pth
import argparse
from datetime import datetime
import pdb
import pickle # loading the data
import gzip
import torch
from torch.optim.lr_scheduler import StepLR
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data   #dataloader
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import seaborn as sns
import json
import datetime as datetime
import time
from pathlib import Path
from os import getcwd, chdir
import glob, os, sys, re
from sklearn.metrics import f1_score
from models_cifar import Resnet, BasicBlockBN
from earlystop import EarlyStopping
from utils_cifar import get_loaders
from train_loop_cifar import train_model, test_model

#*------------------------------------------------------------*

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#*------------------------------------------------------------*

time_now = datetime.now().strftime("%d%b%Y_%H%M%S")
default_path = os.path.join(Path().cwd(), time_now)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
r = len(classes)
num_classes = r

parser = argparse.ArgumentParser(description = 'Argument parser to automate experiments-running process.')
parser.add_argument('-norm', '--normalization', type = str, default = 'torch_bn', action = 'store')  # [ bn | in | bin | ln | gn | nn | torch_bn] 
parser.add_argument('-n', '--n', type = int, default = 2, action = 'store') # [1 |  2 | 3 ] 
parser.add_argument('-d', '--data_dir', type = str, default = './data', action = 'store')
parser.add_argument('-f', '--output_file', type = str, default = default_path, action = 'store')
args = parser.parse_args()

norm_key = args.normalization
n = args.n
data_dir = args.data_dir
checkpt_folder = args.output_file

FOLDER = os.getcwd()
pr = f"n = {n} |  r = {r}  | epochs = {epochs} | lr = {lr} | batch_size = {batch_size} | NL = {norm_ley}"
print(pr)

if device == 'cuda':
    torch.cuda.empty_cache() 

train_loader, val_loader, test_loader = get_loaders(data_dir, batch_size, transform_train, transform_test)
model =  Resnet(BasicBlockBN, n_layers = n, num_classes = num_classes, input_dim = 3, norm_layer = norm_key).to(device)
model, train_loss, valid_loss, percentile_1, percentile_20, percentile_80, percentile_99 = train_model(model, train_loader, val_loader, checkpt_folder, norm_key)