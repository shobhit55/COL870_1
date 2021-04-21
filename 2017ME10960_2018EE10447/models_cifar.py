import pdb
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
# from torchvision.utils import save_image
import numpy as np
import pandas as pd
# import networkx
# from tqdm import tqdm
# import json
# import datetime as datetime
# import time
# from pathlib import Path
# from os import getcwd, chdir
# import glob, os, sys, re
# import cv2
# from PIL import Image
# from sklearn.metrics import f1_score
# print(f"Pytorch version: {torch.__version__}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

G  =  4
batch_size = 128

class BN(nn.Module):
  def __init__(self, input_dim, mom=0.9, eps=1e-5): #norm_layer
      super().__init__()
      self.input_dim = input_dim
      self.mom = mom
      self.eps = eps
      self.gamma = nn.Parameter(torch.ones(1,input_dim,1,1))
      self.beta = nn.Parameter(torch.zeros(1,input_dim,1,1))
      self.moving_mean = torch.zeros(1,input_dim,1,1).to(device)
      self.moving_var = torch.ones(1,input_dim,1,1).to(device)

  def forward(self, x):
      if not torch.is_grad_enabled():
          return self.gamma*(x-self.moving_mean)/torch.sqrt(self.moving_var+self.eps) + self.beta

      xmean = x.mean(dim=(0,2,3), keepdim=True)
      xvar = ((x-xmean)**2).mean(dim=(0,2,3), keepdim=True)
      Xhat = (x - xmean)/torch.sqrt(xvar+self.eps) 
   
      with torch.no_grad():
        
        self.moving_mean = self.mom*(self.moving_mean) + (1-self.mom)*xmean
        self.moving_var = self.mom*(self.moving_var) + (1-self.mom)*xvar

      return self.gamma*Xhat + self.beta


class IN(nn.Module):

    def __init__(self, input_dim, eps=1e-5, mom=0.9, batch_size = batch_size): #norm_layer
        super().__init__()
        self.input_dim = input_dim
        self.G = input_dim
        self.mom = mom
        self.eps = eps
        self.batch_size = batch_size
        self.moving_mean = torch.zeros(batch_size, self.G, 1, 1, 1).to(device)
        self.moving_var = torch.ones(batch_size, self.G, 1, 1, 1).to(device)
        self.gamma = nn.Parameter(torch.ones(1,input_dim,1,1))
        self.beta = nn.Parameter(torch.zeros(1,input_dim,1,1))

    def forward(self, x):
        N, C, H, W = x.shape 
        if not torch.is_grad_enabled():
            x = x.view(N, self.G, C // self.G, H, W)
            x = (x - self.moving_mean)/torch.sqrt(self.moving_var+self.eps) 
            x = x.view(N, C, H, W)
            return x*self.gamma+ self.beta

        x = x.view(N, self.G, C // self.G, H, W)
        xmean = x.mean(dim = [2, 3, 4], keepdim = True).to(device)
        xvar = torch.mean( (x-xmean)**2, [2, 3, 4], keepdim = True).to(device)
        x = (x - xmean) / torch.sqrt(xvar + self.eps)
        x = x.view(N, C, H, W) 

        with torch.no_grad():
          self.moving_mean = self.mom*(self.moving_mean) + (1-self.mom)*xmean
          self.moving_var = self.mom*(self.moving_var) + (1-self.mom)*xvar
        return x*self.gamma + self.beta


class BIN(nn.Module):
    def __init__(self, input_dim, mom=0.9, eps=1e-5, batch_size = batch_size): #norm_layer
        super().__init__()
        self.input_dim = input_dim
        self.mom = mom
        self.eps = eps
        self.batch_size = batch_size
        self.gamma = nn.Parameter(torch.ones(1,input_dim,1,1))
        self.beta = nn.Parameter(torch.zeros(1,input_dim,1,1))
        self.rho = nn.Parameter(torch.tensor(0.5))
        self.moving_meanbn = torch.zeros(1,input_dim,1,1).to(device)
        self.moving_varbn = torch.ones(1,input_dim,1,1).to(device)
        self.moving_meanin = torch.zeros(batch_size, input_dim,1,1).to(device)
        self.moving_varin = torch.ones(batch_size, input_dim,1,1).to(device)

    def forward(self, x):
        if not torch.is_grad_enabled():
            Xhatbn = (x - self.moving_meanbn)/torch.sqrt(self.moving_varbn + self.eps) #BN
            Xhatin = (x - self.moving_meanin)/torch.sqrt(self.moving_varin + self.eps) #BN
            return self.gamma*(self.rho*Xhatbn + (1-self.rho)*Xhatin) + self.beta
        
        xmeanbn = x.mean(dim=(0,2,3), keepdim=True) #BN
        xvarbn = ((x-xmeanbn)**2).mean(dim=(0,2,3), keepdim=True) #BN
        Xhatbn = (x - xmeanbn)/torch.sqrt(xvarbn+self.eps) #BN

        xmeanin = x.mean(dim=(2,3), keepdim=True) #IN
        xvarin = ((x-xmeanin)**2).mean(dim=(2,3), keepdim=True) #IN
        Xhatin = (x - xmeanin)/torch.sqrt(xvarin+self.eps) #IN
        
        Xhat = self.rho*Xhatbn + (1-self.rho)*Xhatin
        #how to keep rho between 0 and 1
        with torch.no_grad():
          self.moving_meanbn = self.mom*(self.moving_meanbn) + (1-self.mom)*xmeanbn
          self.moving_varbn = self.mom*(self.moving_varbn) + (1-self.mom)*xvarbn

          self.moving_meanin = self.mom*(self.moving_meanin) + (1-self.mom)*xmeanin
          self.moving_varin = self.mom*(self.moving_varin) + (1-self.mom)*xvarin
        return self.gamma*Xhat + self.beta


class LN(nn.Module):

    def __init__(self, input_dim, eps=1e-5, mom=0.9, batch_size = batch_size): #norm_layer
        super().__init__()
        self.input_dim = input_dim
        self.G = 1
        self.mom = mom
        self.eps = eps
        self.batch_size = batch_size
        self.moving_mean = torch.zeros(batch_size, self.G, 1, 1, 1).to(device)
        self.moving_var = torch.ones(batch_size, self.G, 1, 1, 1).to(device)
        self.gamma = nn.Parameter(torch.ones(1,input_dim,1,1))
        self.beta = nn.Parameter(torch.zeros(1,input_dim,1,1))

    def forward(self, x):
        N, C, H, W = x.shape 
        if not torch.is_grad_enabled():
            x = x.view(N, self.G, C // self.G, H, W)
            x = (x - self.moving_mean)/torch.sqrt(self.moving_var+self.eps) 
            x = x.view(N, C, H, W)
            return x*self.gamma+ self.beta
        x = x.view(N, self.G, C // self.G, H, W)
        xmean = x.mean(dim = [2, 3, 4], keepdim = True).to(device)
        xvar = torch.mean( (x-xmean)**2, [2, 3, 4], keepdim = True).to(device)
        x = (x - xmean) / torch.sqrt(xvar + self.eps)
        x = x.view(N, C, H, W) 
        with torch.no_grad():
          self.moving_mean = self.mom*(self.moving_mean) + (1-self.mom)*xmean
          self.moving_var = self.mom*(self.moving_var) + (1-self.mom)*xvar
        return x*self.gamma + self.beta


class GN(nn.Module):
    def __init__(self, input_dim, G = 4, eps=1e-5, mom=0.9, batch_size = batch_size): #norm_layer
        super().__init__()
        self.input_dim = input_dim
        self.G = G
        self.mom = mom
        self.eps = eps
        self.batch_size = batch_size
        self.moving_mean = torch.zeros(batch_size, self.G, 1, 1, 1).to(device)
        self.moving_var = torch.ones(batch_size, self.G, 1, 1, 1).to(device)
        self.gamma = nn.Parameter(torch.ones(1,input_dim,1,1))
        self.beta = nn.Parameter(torch.zeros(1,input_dim,1,1))

    def forward(self, x):
        N, C, H, W = x.shape 
        if not torch.is_grad_enabled():
            x = x.view(N, self.G, C // self.G, H, W)
            x = (x - self.moving_mean)/torch.sqrt(self.moving_var+self.eps) 
            x = x.view(N, C, H, W)
            return x*self.gamma+ self.beta

        x = x.view(N, self.G, C // self.G, H, W)
        xmean = x.mean(dim = [2, 3, 4], keepdim = True).to(device)
        xvar = torch.mean( (x-xmean)**2, [2, 3, 4], keepdim = True).to(device)
        x = (x - xmean) / torch.sqrt(xvar + self.eps)
        x = x.view(N, C, H, W) 

        with torch.no_grad():
          self.moving_mean = self.mom*(self.moving_mean) + (1-self.mom)*xmean
          self.moving_var = self.mom*(self.moving_var) + (1-self.mom)*xvar
        return x*self.gamma + self.beta


class BasicBlockBN(nn.Module):
    def __init__(self, input_dim, output_dim, identity_downsample = None, stride=1, norm_layer = 'torch_bn'):
        super(BasicBlockBN, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1, bias = False)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1, bias = False)
        self.norm_layer = norm_layer
        # print("block")
        if self.norm_layer!='nn':
          if self.norm_layer == 'torch_bn':
            # print("in bn")
            self.bn1 = nn.BatchNorm2d(output_dim).to(device)
            self.bn2 = nn.BatchNorm2d(output_dim).to(device)
          else:
            if self.norm_layer=="bn":
              # print(self.norm_layer)
              self.bn1 = BN(input_dim=output_dim).to(device)
              self.bn2= BN(input_dim=output_dim).to(device)
            elif self.norm_layer=="in":
              # print(self.norm_layer)
              self.bn1 = IN(input_dim=output_dim).to(device)
              self.bn2 = IN(input_dim=output_dim).to(device)
            elif self.norm_layer=="bin":
              # print(self.norm_layer)
              self.bn1 = BIN(input_dim=output_dim).to(device)
              self.bn2 = BIN(input_dim=output_dim).to(device)
            elif self.norm_layer=="ln":
              # print(self.norm_layer)
              self.bn1 = LN(input_dim=output_dim).to(device)
              self.bn2 = LN(input_dim=output_dim).to(device)
            elif self.norm_layer=="gn":
              # print(self.norm_layer)
              self.bn1 = GN(input_dim=output_dim,G=G).to(device)
              self.bn2 = GN(input_dim=output_dim,G=G).to(device)
    
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.init_weights()

    def init_weights(self):
      for m in self.modules():
        if isinstance(m, nn.Conv2d):
          # nn.init.kaiming_uniform_(m.weight)
          nn.init.xavier_uniform_(m.weight, gain = 5)
        if isinstance(m, nn.Linear):
          nn.init.kaiming_uniform_(m.weight)
          # nn.init.constant_(m.bias,0)
      
    def forward(self, x):
        input = x.to(device)
        x = self.conv1(x)
        if self.norm_layer!='nn':
          x = self.bn1(x).to(device)
        x = self.relu(x)
        x = self.conv2(x)
        if self.norm_layer!='nn':
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
        # print("R")
        if self.norm_layer!='nn':
          if self.norm_layer == 'torch_bn':
            # print("in bn")
            self.bn1 = nn.BatchNorm2d(16).to(device)
          else:
            if self.norm_layer=="bn":
              # print(self.norm_layer)
              self.bn1 = BN(input_dim=16).to(device)
            elif self.norm_layer=="in":
              # print(self.norm_layer)
              self.bn1 = IN(input_dim=16).to(device)
            elif self.norm_layer=="bin":
              # print(self.norm_layer)
              self.bn1 = BIN(input_dim=16).to(device)
            elif self.norm_layer=="ln":
              # print(self.norm_layer)
              self.bn1 = LN(input_dim=16).to(device)
            elif self.norm_layer=="gn":
              # print(self.norm_layer)
              self.bn1 = GN(input_dim=16, G=G).to(device)

        self.relu1 = nn.ReLU(inplace = True)
        self.layer1 = self.layer(16, 16, block, n_layers, stride=1) #32x32 output, 16channels
        self.layer2 = self.layer(16, 32, block, n_layers) #16x16 output, 32channels
        self.layer3 = self.layer(32, 64, block, n_layers) #8x8 output, 64channels
        self.pool_out = nn.AvgPool2d(kernel_size=8) #1x1 output, 64 channels
        self.fc_out_layer = nn.Linear(64,num_classes) # fully connected output layer
        self.init_weights()
        self.fea = None

    def layer(self, input_dim, output_dim, block, num_blocks, stride=2):
        # print("layer")
        if self.norm_layer!='nn':
          if self.norm_layer == 'torch_bn':
            # print("in bn")
            bn = nn.BatchNorm2d(output_dim).to(device)
          else:
            if self.norm_layer =="bn":
              # print(self.norm_layer)
              bn = BN(input_dim=output_dim).to(device)
            elif self.norm_layer =="in":
              # print(self.norm_layer)
              bn = IN(input_dim=output_dim).to(device)
            elif self.norm_layer =="bin":
              # print(self.norm_layer)
              bn = BIN(input_dim=output_dim).to(device)
            elif self.norm_layer =="ln":
              # print(self.norm_layer)
              bn = LN(input_dim=output_dim).to(device)
            elif self.norm_layer =="gn":
              # print(self.norm_layer)
              bn = GN(input_dim=output_dim,G=G).to(device)

        if stride!=1:
          cov = nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=2, bias =False)
          nn.init.kaiming_uniform_(cov.weight)
          if self.norm_layer!='nn':
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
          # nn.init.kaiming_uniform_(m.weight)
          nn.init.xavier_uniform_(m.weight, gain = 2)


    def pr(self):
      for m in self.modules():
        if isinstance(m, nn.Conv2d):
          continue
          # print(m.weight)
        if isinstance(m, nn.Linear):
          continue
          # print(m.weight)
    
    def get_fea(self):
      return self.fea

    def forward(self, x):
        x = self.conv1(x)
        if self.norm_layer!='nn':
          x = self.bn1(x).to(device)          
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # self.fea = self.fea.detach().cpu()
        x = self.pool_out(x).to(device)
        x = x.view(-1,64)
        self.fea = x.clone().detach().cpu()
        # pdb.set_trace()
        x = self.fc_out_layer(x)
        return x