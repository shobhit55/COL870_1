from dataset_ner import NERGmbData
from dict_save_ner import glove_vocab_save, train_vocab_save, pre_trained_tens
from models_ner import BiLSTM_crf, BiLSTM
from train_loops_ner import train_model_crf, train_model
from utils_ner import get_loaders
import pickle
import torch
import sys

argslist = sys.argv[1:]

i=0
while i<len(argslist):
    if argslist[i]=='--initialization':
        if argslist[i+1]=='random':
            pre_tr = False
        else:
            pre_tr = True
    
    if argslist[i]=='--char_embeddings':
        if argslist[i+1]=='0':
            char_level = False
        else:
            char_level = True

    if argslist[i]=='--layer_normalization':
        if argslist[i+1]=='0':
            norm = False
        else:
            norm = True

    if argslist[i]=='--crf':
        if argslist[i+1]=='0':
            crf = False
        else:
            crf = True

    if argslist[i]=='--output_file':
        model_path = argslist[i+1]

    if argslist[i]=='--data_dir':
        data_dir = argslist[i+1]

    if argslist[i]=='--glove_embeddings_file':
        glove_file = argslist[i+1]

    if argslist[i]=='--vocabulary_output_file':
        vocab_path = argslist[i+1]

    if argslist[i]=='--vocabulary_output_file':
        vocab_path = argslist[i+1]

    i+=2
'''
--initialization [random | glove ] --char_embeddings [ 0 | 1 ] --layer_normalization [ 0 | 1 ] --crf [ 0 | 1 ]
 --output_file <path to the trained model> --data_dir <directory containing data> 
 --glove_embeddings_file <path to file containing glove embeddings> --vocabulary_output_file <path to the file in which vocabulary will be written>
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------- inputs ------------------

pre_tr = True #run data header also, if this changed
norm = False
crf = True
char_level = False
dropout = 0.5
epochs = 100

train_file = data_dir + '/train.txt'
val_file = data_dir + '/dev.txt'

batch_size = 128
no_fine_tune = False
input_size = 100 + 50*char_level
patience = 4
lr = 0.001
optim_key = 'Adam' #'AdaDel' #'SGD', Adam for dropout
# ----------------------------------

# ----------------- dicts load ---------------------
tag_idx = {'O':0, 'B-tim':1, 'I-tim':2, 'B-gpe':3, 'I-gpe':4, 'B-eve':5, 'I-eve':6, 'B-per':7, 'I-per':8, 'B-art':9, 'I-art':10, 'B-org':11, 'I-org':12, 'I-geo':13, 'B-geo':14, 'B-nat':15, 'I-nat':16}
tag_list = ['O', 'B-tim', 'I-tim', 'B-gpe', 'I-gpe', 'B-eve', 'I-eve', 'B-per', 'I-per', 'B-art', 'I-art', 'B-org', 'I-org', 'I-geo', 'B-geo', 'B-nat', 'I-nat']
num_tags = len(tag_list)

if pre_tr:
    glove_vocab_save(glove_file, train_file, vocab_path)
    pre_trained = pre_trained_tens(glove_file)
else:
    train_vocab_save(train_file, vocab_path)
    pre_trained = None
# dicts load
a_file = open(vocab_path,"rb")
vocab = pickle.load(a_file)
word_idx = vocab['word_idx']
char_idx = vocab['char_idx']
vocab_size = len(word_idx) 
char_size = len(char_idx)
#------------------------------------------------------------------

torch.cuda.empty_cache()
train_loader, val_loader = get_loaders(batch_size, file_path = train_file, word_idx = word_idx, tag_idx = tag_idx, char_idx = char_idx), get_loaders(batch_size, file_path = val_file, word_idx = word_idx, tag_idx = tag_idx, char_idx = char_idx)
if crf:
    use_hidden_layer = True
    model = BiLSTM_crf(pre_trained=pre_trained, use_hidden_layer = use_hidden_layer, vocab_size=vocab_size, char_size=char_size).to(device)
    model = train_model_crf(model, train_loader, val_loader, device=device, path = model_path, optim_key=optim_key, epochs=epochs)
else:
    model = BiLSTM(pre_trained, input_size, num_tags, dropout=dropout, pre_tr=pre_tr, norm=norm, char_level=char_level, char_size=char_size, vocab_size=vocab_size).to(device)
    model, train_loss, valid_loss, train_ac, val_ac, f1_mi_train, f1_ma_train, f1_mi_val, f1_ma_val = train_model(model, train_loader, val_loader, device = device, path = model_path, optim_key=optim_key, epochs=epochs)