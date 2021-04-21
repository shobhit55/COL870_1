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
from earlystop import EarlyStopping
print(f"Pytorch version: {torch.__version__}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

epochs = 100
patience = 30
lr = 0.1
optim_key = "SGD"
criterion = nn.CrossEntropyLoss()
batch_size = 128
momentum = 0.9
start_epoch = 1
wd = 1e-4

def test_model(model, test_loader):
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    correct = 0
    l1 = []
    l2 = []
    model.eval()
    for data, target in test_loader:
        if data.shape[0]!=batch_size:
            break
        data, target = data.to(device), target.to(device)
        output = model(data).to(device)
        loss = criterion(output, target)
        test_loss += loss.item()*data.size(0)
        _, pred = torch.max(output, 1)
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        # correct += (pred == target).sum().cpu()
        l1.extend(target.data.cpu())
        l2.extend(pred.data.cpu())
        for i in range(len(target.data)):
            # print(i, target.data[i])
            l = target.data[i]
            class_correct[l] += correct[i].item()
            class_total[l] += 1
            # precision[l] =
            # recall[l] = 

    test_loss = test_loss/len(test_loader.dataset)
    save_print(f'Test Loss: {test_loss}',logger)
    # print('Acc',correct/len(test_loader.dataset))
    # micro = get_microF1(np.asarray(l1),np.asarray(l2))
    # macro = get_macroF1(np.asarray(l1),np.asarray(l2))
    for i in range(10):
        if class_total[i] > 0:
            save_print(f'Test Accuracy of {str(i), 100 * class_correct[i] / class_total[i]}: ({np.sum(class_correct[i])}/{np.sum(class_total[i])})', logger)
        else:
            save_print(f'Test Accuracy of {(classes[i])}: N/A (no training examples)', logger)

    t = 100. * np.sum(class_correct) / np.sum(class_total)
    save_print(f'Test Accuracy (Overall): {t} {np.sum(class_correct)}/{np.sum(class_total)}',logger)
    
    return l1,l2

def train_model(model, train_loader, val_loader, checkpt_folder, key):
    print("Training Started...")    
    pr = f"patience = {patience} | epochs = {epochs} | lr = {lr} | momentum = {momentum} | wd = {wd} | batch_size = {batch_size}"
    print(pr)
    optimizer_dict = { "Adam": torch.optim.Adam(model.parameters(), lr = lr), "SGD": torch.optim.SGD(model.parameters(), lr = lr) }  
    
    if key == 'nn':
        optimizer = torch.optim.Adadelta(model.parameters(), lr = lr, rho = 0.6, eps = 1e-06, weight_decay=wd)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum)
    
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    if device  == 'cuda':
        torch.cuda.synchronize()
    acc = []
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = [] 
    test_acc = 0
    start_epoch = 1
    percentile_1=[]
    percentile_20=[]
    percentile_80=[]
    percentile_99=[]
    early_stopping = EarlyStopping(patience=patience, path = checkpt_folder, verbose=True)
    for epoch in range(start_epoch, epochs+1):
      since = time.time()
      train_loss = 0
      correct = 0
      total = 0
      fea_list = []
      model.train()
      for batch, (input, target) in enumerate(train_loader):
          if input.shape[0]!=batch_size:
            break
          input, target = input.to(device), target.to(device) 
          optimizer.zero_grad()
          output = model(input).to(device)
          loss = criterion(output, target)
          loss.backward()
          optimizer.step()
          train_losses.append(loss.item())

      model.eval()
      for input, target in val_loader:
          if input.shape[0]!=batch_size:
              break
          input, target = input.to(device), target.to(device) 
          output = model(input).to(device)
          fea = model.get_fea().view(-1,1).numpy()
          fea_list.append(list(fea))
          loss = criterion(output, target)
          valid_losses.append(loss.item())

      train_loss = np.average(train_losses)
      valid_loss = np.average(valid_losses)
      avg_train_losses.append(train_loss)
      avg_valid_losses.append(valid_loss)
      epoch_len = len(str(epochs))
      percentile_1.append(np.percentile(fea_list,1))
      percentile_20.append(np.percentile(fea_list,20))
      percentile_80.append(np.percentile(fea_list,80))
      percentile_99.append(np.percentile(fea_list,99))
      
      print("-----------------------------------------------------------")
      p_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                    f'train_loss: {train_loss:.5f} ' +
                    f'valid_loss: {valid_loss:.5f}')
      
      print(p_msg)
      train_losses = []
      valid_losses = []
      total_norm = 0
      for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
      total_norm = total_norm ** (1. / 2)
      print("total_gradient_norm:", total_norm)
      early_stopping(valid_loss, model)
      if early_stopping.early_stop:
          print("Early stopping")
          break
      time_elapsed = time.time() - since
      print('Epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
      print()
      scheduler.step()
    
    model.load_state_dict(torch.load(checkpt_folder))
    print("Training Done...")
    return  model, avg_train_losses, avg_valid_losses, percentile_1, percentile_20, percentile_80, percentile_99