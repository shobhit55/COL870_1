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
from torchvision.utils import save_image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import networkx
import seaborn as sns
import plotly.offline
from tqdm import tqdm
import json
import datetime as datetime
import time
from pathlib import Path
from os import getcwd, chdir
import glob, os, sys, re
import cv2
from PIL import Image
from sklearn.metrics import f1_score
print(f"Pytorch version: {torch.__version__}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from normlayer import BN,IN,BIN,LN,GN

class BasicBlockBN(nn.Module):
    def __init__(self, input_dim, output_dim, identity_downsample = None, stride=1, norm_layer = None):
        super(BasicBlockBN, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1, bias = False)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1, bias = False)
        print("block")
        if norm_layer != 'nn':
          if norm_layer == None:
            print('torch bn')
            self.bn1 = nn.BatchNorm2d(output_dim).to(device)
            self.bn2 = nn.BatchNorm2d(output_dim).to(device)
          else:
            if norm_layer=="bn":
              print(norm_layer)
              self.bn1 = BN(input_dim=output_dim).to(device)
              self.bn2= BN(input_dim=output_dim).to(device)
            elif norm_layer=="in":
              print(norm_layer)
              self.bn1 = IN(input_dim=output_dim).to(device)
              self.bn2 = IN(input_dim=output_dim).to(device)
            elif norm_layer=="bin":
              print(norm_layer)
              self.bn1 = BIN(input_dim=output_dim).to(device)
              self.bn2 = BIN(input_dim=output_dim).to(device)
            elif norm_layer=="ln":
              print(norm_layer)
              self.bn1 = LN(input_dim=output_dim).to(device)
              self.bn2 = LN(input_dim=output_dim).to(device)
            elif norm_layer=="gn":
              print(norm_layer)
              self.bn1 = GN(input_dim=output_dim,G=G).to(device)
              self.bn2 = GN(input_dim=output_dim,G=G).to(device)
    
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.init_weights()

    def init_weights(self):
      for m in self.modules():
        if isinstance(m, nn.Conv2d):
          nn.init.kaiming_uniform_(m.weight)
        if isinstance(m, nn.Linear):
          nn.init.kaiming_uniform_(m.weight)
          nn.init.constant_(m.bias,0)
      
    def forward(self, x):
        input = x.to(device)
        x = self.conv1(x)
        if norm_layer != 'nn':
          x = self.bn1(x).to(device)
        x = self.relu(x)
        x = self.conv2(x)
        if norm_layer != 'nn':
          x = self.bn2(x).to(device)
        if self.identity_downsample is not None:
            input = self.identity_downsample(input).to(device)
        x = x + input
        x = self.relu(x)
        return x

class Resnet(nn.Module):
    def __init__(self, block, n_layers, num_classes = 10, input_dim = 3, norm_layer = None):
        super(Resnet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 16, kernel_size = 3, stride=1, padding=1, bias = False) #32x32 output # Random crop left
        self.norm_layer = norm_layer
        print("R")
        if norm_layer != 'nn':
          if norm_layer == None:
            print("torch bn")
            self.bn1 = nn.BatchNorm2d(16).to(device)
          else:
            if norm_layer=="BN":
              print(self.norm_layer)
              self.bn1 = BN(input_dim=16).to(device)
            elif norm_layer=="IN":
              print(self.norm_layer)
              self.bn1 = IN(input_dim=16).to(device)
            elif norm_layer=="BIN":
              print(self.norm_layer)
              self.bn1 = BIN(input_dim=16).to(device)
            elif norm_layer=="LN":
              print(self.norm_layer)
              self.bn1 = LN(input_dim=16).to(device)
            elif norm_layer=="GN":
              print(self.norm_layer)
              self.bn1 = GN(input_dim=16, G=G).to(device)

        self.relu1 = nn.ReLU(inplace = True)
        self.layer1 = self.layer(16, 16, block, n_layers, stride=1) #32x32 output, 16channels
        self.layer2 = self.layer(16, 32, block, n_layers) #16x16 output, 32channels
        self.layer3 = self.layer(32, 64, block, n_layers) #8x8 output, 64channels
        self.pool_out = nn.AvgPool2d(kernel_size=8) #1x1 output, 64 channels
        self.fc_out_layer = nn.Linear(64,num_classes) # fully connected output layer
        # self.init_weights()
        self.fea = None

    def layer(self, input_dim, output_dim, block, num_blocks, stride=2):
        print("layer")
        if use_bn:

          if self.norm_layer != 'nn':
            print("torch bn")
            bn = nn.BatchNorm2d(output_dim).to(device)
          else:
            if self.norm_layer =="bn":
              print(self.norm_layer)
              bn = BN(input_dim=output_dim).to(device)
            elif self.norm_layer =="in":
              print(self.norm_layer)
              bn = IN(input_dim=output_dim).to(device)
            elif self.norm_layer =="bin":
              print(self.norm_layer)
              bn = BIN(input_dim=output_dim).to(device)
            elif self.norm_layer =="ln":
              print(self.norm_layer)
              bn = LN(input_dim=output_dim).to(device)
            elif self.norm_layer =="gn":
              print(self.norm_layer)
              bn = GN(input_dim=output_dim,G=G).to(device)

        if stride!=1:
          cov = nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=2, bias =False)
          nn.init.kaiming_uniform_(cov.weight)
          if norm_layer != 'nn':
            identity_downsample_1 = nn.Sequential(
                                                    cov,
                                                    bn
                                                  ).to(device)
          else:
            identity_downsample_1 = nn.Sequential(
              cov
            ).to(device)             

        else:
            identity_downsample_1 = None

        layers = []
        layers.append(block(input_dim, output_dim, identity_downsample_1, stride, norm_layer = self.norm_layer ).to(device)) #increases channels and downsamples feature map
                            # input_dim, output_dim, identity_downsample = None, stride=1, norm_layer = None):
        for i in range(num_blocks-1):
            layers.append(block(output_dim, output_dim, norm_layer = self.norm_layer).to(device))
        return nn.Sequential(*layers)

    def init_weights(self):
      for m in self.modules():
        if isinstance(m, nn.Conv2d):
          nn.init.kaiming_uniform_(m.weight)
        if isinstance(m, nn.Linear):
          nn.init.kaiming_uniform_(m.weight)
          nn.init.constant_(m.bias,0)

    def pr(self):
      for m in self.modules():
        if isinstance(m, nn.Conv2d):
          print(m.weight)
        if isinstance(m, nn.Linear):
          print(m.weight)
    def get_fea(self):
      return self.fea
    def forward(self, x):

        x = x.to(device)
        x = self.conv1(x)
        if norm_layer != 'nn':
          x = self.bn1(x).to(device)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.pool_out(x).to(device)
        x = x.view(-1,64)
        self.fea = x.clone().detach().cpu()
        
        x = self.fc_out_layer(x)
        return x