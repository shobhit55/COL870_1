import pandas as pd
import pickle
import csv
import torch

def glove_vocab_save(glove_file, train_file, vocab_path):
    data = pd.read_csv(glove_file, sep=" ", header=None, quoting=csv.QUOTE_NONE)
    data_file = open(train_file).read()
    idx = {}
    idx['Nil']=0
    idx['Unk']=1
    chr_idx={}
    chr_idx['Nil']=0
    chr_idx['Unk']=1
    for i in range(data.count()[0]):
        idx[data.iloc[i][0]] = i+2 #index 0 reserved for no word, 1 reserved for out of vocab word
    for i, char in enumerate(set(data_file)):
        chr_idx[char] = i+2
    vocab = {'word_idx':idx, 'char_idx':chr_idx}

    a_file = open(vocab_path, "wb")
    pickle.dump(vocab, a_file)
    a_file.close()

def pre_trained_tens(glove_file):
    data = pd.read_csv(glove_file, sep=" ", header=None, quoting=csv.QUOTE_NONE)
    tensor = torch.zeros(data.count()[0]+2, 100) #0th row represents 'no word'
    for i in range(data.count()[0]):
        tensor[i+2,:] = torch.tensor(list(map(float, data.iloc[i][1:])))
    return tensor

def train_vocab_save(train_file, vocab_path):
    data_file = open(train_file).read()
    train_data = pd.read_csv(train_file, sep=" ", header=None, encoding="latin1").dropna()
    train_data.columns = ["Token", "POS Tag", "Word", "NER Tag"]
    train_data_token_set = set(train_data['Token'].tolist())
    train_data_token_dict={}
    train_data_token_dict['Nil']=0
    train_data_token_dict['Unk']=1
    i=0
    for x in train_data_token_set:
        train_data_token_dict[x] = i+2
        i+=1
    
    chr_idx={}
    chr_idx['Nil']=0
    chr_idx['Unk']=1
    for i, char in enumerate(set(data_file)):
        chr_idx[char] = i+2
    
    vocab = {}
    vocab['word_idx':train_data_token_dict, 'char_idx':chr_idx]
    a_file = open(vocab_path, "wb")
    pickle.dump(vocab, a_file)
    a_file.close()