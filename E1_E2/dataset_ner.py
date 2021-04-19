import torch
from torch.utils.data import Dataset, DataLoader

# dataset
def word2ind(word, word_idx, char_idx):
    ind = [word_idx[word] if word in word_idx.keys() else 1] #1
    for char in word:
        if char in char_idx.keys():
            a = char_idx[char]
        else:
            a = 1
        ind.append(a)
    return ind

class NERGmbData(Dataset):
    def __init__(self, csv_file, word_idx, char_idx, tag_idx, transform=None):
        self.data_file = open(csv_file).read().split('\n\n')[:-1] # deleting the last empty sentence, pd.read_csv(csv_file, sep=" ", header=None, encoding="latin1").dropna()
        # self.root_dir = root_dir
        self.word_idx = word_idx
        self.char_idx = char_idx
        self.tag_idx = tag_idx
        self.transform = transform
    
    def __len__(self):
        return len(self.data_file)
    
    def __getitem__(self, idx): #return words and tags indices
        sent = self.data_file[idx].split('\n') #list of words with tags
        sent_ind = [word2ind(w.split(' ')[0], self.word_idx, self.char_idx) for w in sent if w!='']
        sent_tag_ind = [self.tag_idx[w.split(' ')[3]] for w in sent if w!='']
        
        return sent_ind, sent_tag_ind

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    max_len = max([len(x) for x in xx])
    max_word_len = max([max([len(y) for y in x]) for x in xx])
    xx_pad = torch.zeros(len(xx), max_len, max_word_len+1, dtype=int)
    yy_pad = -torch.ones(len(xx), max_len, dtype=int)
    
    for i, (s,t) in enumerate(zip(xx,yy)):
        for j, w in enumerate(s):
            xx_pad[i,j,:len(w)] = torch.tensor(w)
        yy_pad[i,:len(s)] = torch.tensor(t)

    return xx_pad, yy_pad #, x_lens, y_lens

class NERGmbData_test(Dataset):
    def __init__(self, csv_file, word_idx, char_idx, tag_idx, transform=None):
        self.data_file = open(csv_file).read().split('\n\n')[:-1] # deleting the last empty sentence, pd.read_csv(csv_file, sep=" ", header=None, encoding="latin1").dropna()
        # self.root_dir = root_dir
        self.word_idx = word_idx
        self.char_idx = char_idx
        self.tag_idx = tag_idx
        self.transform = transform
    
    def __len__(self):
        return len(self.data_file)
    
    def __getitem__(self, idx): #return words and tags indices
        sent = self.data_file[idx].split('\n') #list of words with tags
        sent_ind = [word2ind(w.split(' ')[0], self.word_idx, self.char_idx) for w in sent if w!='']
        sent_tag_ind = [self.tag_idx[w.split(' ')[3]] for w in sent if w!='']
        
        return sent_ind, sent_tag_ind, idx

def pad_collate_test(batch):
    (xx, yy, inds) = zip(*batch)
    lengths = [len(x) for x in xx]
    max_len = max(lengths)
    max_word_len = max([max([len(y) for y in x]) for x in xx])
    xx_pad = torch.zeros(len(xx), max_len, max_word_len+1, dtype=int)
    yy_pad = -torch.ones(len(xx), max_len, dtype=int)
    
    for i, (s,t) in enumerate(zip(xx,yy)):
        for j, w in enumerate(s):
            xx_pad[i,j,:len(w)] = torch.tensor(w)
        yy_pad[i,:len(s)] = torch.tensor(t)

    return xx_pad, yy_pad, lengths, inds #, x_lens, y_lens