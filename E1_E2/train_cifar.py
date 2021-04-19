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
from model import Resnet, BasicBlockBN
from earlystop import EarlyStopping
# from utils import 

# time_now = datetime.now().strftime("%d%b%Y_%H%M%S")
# default_path = os.path.join(Path().cwd(), time_now)
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# r = len(classes)
# num_classes = r


# parser = argparse.ArgumentParser(description = 'Argument parser to automate experiments-running process.')
# parser.add_argument('-norm', '--normalization', type = str, default = 'torch_bn', action = 'store')  # [ bn | in | bin | ln | gn | nn | torch_bn] 
# parser.add_argument('-n', '--n', type = int, default = 2, action = 'store') # [1 |  2 | 3 ] 
# parser.add_argument('-d', '--data_dir', type = str, default = '', action = 'store')
# parser.add_argument('-f', '--output_file', type = str, default = default_path, action = 'store')
# args = parser.parse_args()

# norm_key = args.norm
# n = args.n
# data_dir = args.data_dir
# FOLDER = os.getcwd()
# logger = open(os.path.join(FOLDER, 'log.txt'), 'w+')
# checkpt_folder = args.f


# epochs = 100
# patience = 30
# lr = 0.1
# optim_key = "Adam"
# criterion = nn.CrossEntropyLoss()
# G = 16
# batch_size = 128
# momentum = 0.9

# start_epoch = 1



# pr = f"n = {n} |  r = {r}  | epochs = {epochs} | lr = {lr} | optim_key = {optim_key} | criterion = cross-entropy | batch_size = {batch_size} | NL = {NL}"
# save_print(pr, logger)

# if device == 'cuda':
#     torch.cuda.empty_cache() 

# train_loader, val_loader, test_loader = get_loaders(data_dir, batch_size, transform_train, transform_test) # data_dir

# model =  Resnet(BasicBlockBN, n_layers = n, num_classes = num_classes, input_dim = 3, norm_layer = norm_key).to(device)
# model, train_loss, valid_loss, percentile_1, percentile_20, percentile_80, percentile_99 = train_model(model, train_loader, val_loader)

# plot_errors(train_loss, valid_loss)
# plot_percentile(percentile_1, percentile_20, percentile_80, percentile_99)
# save_print("train",logger)
# a,b = test_model(model, train_loader)
# micro = get_microF1(a,b)
# macro = get_macroF1(a,b)
# save_print(f"Macro f1:{macro}, Micro f1:{micro}", logger)
# save_print("val",logger)
# a,b = test_model(model, val_loader)
# micro = get_microF1(a,b)
# macro = get_macroF1(a,b)
# save_print(f"Macro f1:{macro}, Micro f1:{micro}", logger)
# save_print("test",logger)
# a,b = test_model(model, test_loader)
# micro = get_microF1(a,b)
# macro = get_macroF1(a,b)
# save_print(f"Macro f1:{macro}, Micro f1:{micro}", logger)
# with open(FOLDER + "/train_loss.txt", "wb") as fp:   #Pickling
#    pickle.dump(train_loss, fp)
# with open(FOLDER + "/val_loss.txt", "wb") as fp:   #Pickling
#    pickle.dump(valid_loss, fp)

# logger.close()
a_file = open("word_idx_from_glove.pkl","rb")
word_idx = pickle.load(a_file)
print("scuess")