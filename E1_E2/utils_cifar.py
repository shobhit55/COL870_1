# import pdb
import pickle # loading the data
# import gzip
import torch
import torchvision
# import matplotlib.pyplot as plt
import torch.nn as nn
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data   #dataloader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import pandas as pd
from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# import json
# import datetime as datetime
# import time
# from pathlib import Path
# from os import getcwd, chdir
# import glob, os, sys, re


def get_accuracy(actual, probs, a=1):
    if not isinstance(actual, torch.Tensor):
      actual = torch.tensor(actual)
    if not isinstance(probs, torch.Tensor):
      probs = torch.tensor(probs)
    preds = torch.argmax(probs, dim=1)
    correct = preds == actual.long()
    accuracy = correct.sum().item()
    if a:
      return accuracy*100/actual.shape[0]
    else: 
      return accuracy*100

def get_macroF1(actual, probs):
    return f1_score(actual, probs, average='macro')
    
def get_microF1(actual, probs):
   return f1_score(actual, probs, average='micro')

def get_loaders(data_dir, batch_size, transform_train, transform_test):
  print("Getting DataLoaders...")
  cifar_train = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
  cifar_train, cifar_val= torch.utils.data.random_split(cifar_train, [40000, 10000]) # change
  # cifar_val = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
  cifar_test = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform_test)
  # a,b = cifar_train.__getitem__(0)
  # print(np.asarray(a))

  cifar_train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=batch_size, shuffle=True, num_workers=2)
  cifar_val_loader = torch.utils.data.DataLoader(cifar_val, batch_size=batch_size, shuffle=True, num_workers=2)
  cifar_test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=batch_size, shuffle=True, num_workers=2)
  
  return  cifar_train_loader, cifar_val_loader , cifar_test_loader 

class dataset_test(data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_file = pd.read_csv(csv_file, sep=",", header=None)
        self.transform = transform
    
    def __len__(self):
        return len(self.data_file)
    
    def __getitem__(self, index): #return words and tags indices
        x = self.data_file.iloc[index].values.tolist()
        x = np.asarray(x)
        x = np.reshape(x, (3,32,32) )
        
        if self.transform:
            x = Image.fromarray(x.astype(np.uint8).transpose(1,2,0))
            x = self.transform(x)

        return x, index

def get_loader_test(batch_size, transform_test, test_file):
    test_dataset = dataset_test(test_file, transform_test)
    cifar_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return cifar_test_loader

def plot_confusion(confusion, folder, save = True):
  if isinstance(confusion, torch.Tensor):
    confusion = confusion.numpy()
  fig = plt.figure()
  ax = fig.add_subplot(111)
  cax = ax.matshow(confusion)
  fig.colorbar(cax)
  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
  plt.savefig(folder)
  plt.show()

def train_val_earlystop():
  # visualize the loss as the network trained
  fig = plt.figure(figsize=(10,8))
  plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
  plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')
  # find position of lowest validation loss
  minposs = valid_loss.index(min(valid_loss))+1 
  plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
  plt.xlabel('epochs')
  plt.ylabel('loss')
  plt.ylim(0, 0.5) # consistent scale
  plt.xlim(0, len(train_loss)+1) # consistent scale
  plt.grid(True)
  plt.legend()
  plt.tight_layout()
  plt.show()
  fig.savefig('loss_plot.png', bbox_inches='tight')

def save_print(strr, logger):
  print(strr)
  logger.write(strr + '\n')
  logger.flush()

def plot_errors(train_loss, valid_loss):# visualize the loss as the network trained
  print("Plotting and saving errors after each epoch...")
  fig = plt.figure(figsize=(10,8))
  plt.plot(range(1,len(train_loss)+1), train_loss, label='Training Loss')
  plt.plot(range(1,len(valid_loss)+1), valid_loss,label='Validation Loss')

  minposs = valid_loss.index(min(valid_loss))+1 
  plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

  plt.xlabel('epochs')
  plt.ylabel('loss')
  # plt.ylim(0, 0.5) # consistent scale
  # plt.xlim(0, len(train_loss)+1) # consistent scale
  plt.grid(True)
  plt.legend()
  plt.tight_layout()
  plt.show()
  fig.savefig(FOLDER + '/loss_plot.png', bbox_inches='tight')

def plot_percentile(l1,l2,l3,l4):
  fig = plt.figure()
  plt.plot(range(1,len(l1)+1), l1, label='1 percentile')
  plt.plot(range(1,len(l2)+1), l2, label='20 percentile')
  plt.plot(range(1,len(l3)+1), l3, label='80 percentile')
  plt.plot(range(1,len(l4)+1), l4, label='99 percentile')

  plt.xlabel('epochs')
  plt.ylabel('percentile')
  # plt.ylim(0, 0.5) # consistent scale
  # plt.xlim(0, len(train_loss)+1) # consistent scale
  plt.grid(True)
  plt.legend()
  plt.tight_layout()
  plt.show()
  fig.savefig(FOLDER + '/percentile.png', bbox_inches='tight')