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
print(f"Pytorch version: {torch.__version__}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path=None, trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        if self.verbose:
            self.trace_func(f'Validation loss decreased {self.val_loss_min:.6f} --> {val_loss:.6f}.  Saving model ...')
        self.val_loss_min = val_loss
        print(self.val_loss_min, self.path)