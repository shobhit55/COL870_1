import argparse
# from datetime import datetime
# import pdb
import pickle # loading the data
# import gzip
import torch
from torch.optim.lr_scheduler import StepLR
import torchvision
import torch.nn as nn
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data   #dataloader
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import os
# import json
# import datetime as datetime
# import time
# from pathlib import Path
# from os import getcwd, chdir
# import glob, os, sys, re
# from sklearn.metrics import f1_score
from models_cifar import Resnet, BasicBlockBN
from earlystop import EarlyStopping
from utils_cifar import get_loaders
from train_loop_cifar import train_model, test_model_output
# print(f"Pytorch version: {torch.__version__}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

default_path = os.getcwd()
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
r = len(classes)
num_classes = r
batch_size = 128

parser = argparse.ArgumentParser(description = 'Argument parser to automate experiments-running process.')
parser.add_argument('-norm', '--normalization', type = str, default = 'torch_bn', action = 'store')  # [ bn | in | bin | ln | gn | nn | torch_bn] 
parser.add_argument('-n', '--n', type = int, default = 2, action = 'store') # [1 |  2 | 3 ] 
parser.add_argument('-f', '--test_data_file', type = str, default = '../data/cifar_test.csv', action = 'store')
parser.add_argument('-j', '--model_file', type = str, default = default_path, action = 'store')
parser.add_argument('-l', '--output_file', type = str, default = default_path, action = 'store')
args = parser.parse_args()

norm_key = args.normalization
n = args.n
test_file = args.test_data_file
model_file = args.model_file
output_file = args.output_file

if device == 'cuda':
    torch.cuda.empty_cache() 

model =  Resnet(BasicBlockBN, n_layers = n, num_classes = num_classes, input_dim = 3, norm_layer = norm_key).to(device)
model.load_state_dict(torch.load(model_file, map_location=device))
test_model_output(model, test_file, output_file, transform_test, batch_size=10)