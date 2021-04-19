import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset_ner import NERGmbData_test, NERGmbData

def criterion(output, target): #batch_size, max_len, num_tags; batch_size, max_len
    output = output.view(-1, output.shape[2]) # reshape the tensor
    target = target.view(-1) # reshape the tensor
    output = output[target!=-1,:] # -1 targets are to be excluded from loss computation
    target = target[target!=-1]
    return -torch.sum(output[range(output.shape[0]),target])/output.shape[0]

def get_loaders(batch_size, file_path, pad_collate):
    dataset = NERGmbData(file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate, num_workers=2, pin_memory=True)

    return dataloader

def get_loader_test(batch_size, file_path, pad_collate_test):
    test_dataset = NERGmbData_test(file_path)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_test, num_workers=2, pin_memory=True)

    return test_dataloader

def accuracy(output, target, tag_ind=None): #batch_size, max_len, num_tags; batch_size, max_len
    output = output.view(output.shape[0]*output.shape[1], output.shape[2]) # reshape the tensor
    target = target.view(target.shape[0]*target.shape[1]) # reshape the tensor
    output = output[target!=-1,:] # -1 -> no words targets are to be excluded
    target = target[target!=-1]

    if tag_ind=='all_O':
        output = output[target!=0,:]
        output = torch.argmax(output, dim=1)
        target = target[target!=0]
    elif tag_ind!=None:
        output = output[target==tag_ind,:]
        output = torch.argmax(output, dim=1)
        target = target[target==tag_ind]
    else:
        output = torch.argmax(output, dim=1)
    n_correct = torch.sum(output==target)
    n_total = output.shape[0]

    return float(n_correct), n_total

class EarlyStopping:
    def __init__(self, path, patience=7, verbose=False, delta=0, trace_func=print):
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
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
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