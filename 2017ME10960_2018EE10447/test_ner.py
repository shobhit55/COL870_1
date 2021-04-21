import torch
import sys
from dataset_ner import NERGmbData_test
from models_ner import BiLSTM_crf, BiLSTM
from train_loops_ner import test_model_crf, test_model_output
from utils_ner import get_loaders
import pickle

'''
--model_file <path to the trained model> --char_embeddings [ 0 | 1 ] --layer_normalization [ 0 | 1 ] --crf [ 0 | 1 ] 
--test_data_file <path to a file in the same format as original train file with random  NER / POS tags for each token> 
--output_file <file in the same format as the test data file with random NER tags replaced with the predictions> 
--glove_embeddings_file <path to file containing glove embeddings> --vocabulary_input_file <path to the vocabulary file written while training>
'''
argslist = sys.argv[1:]
i=0
while i<len(argslist):
    if argslist[i]=='--initialization':
        if argslist[i+1]=='random':
            pre_tr = False
        else:
            pre_tr = True
    
    if argslist[i]=='--model_file':
        model_path = argslist[i+1]

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
    
    if argslist[i]=='--test_data_file':
        test_file = argslist[i+1]
    
    if argslist[i]=='--output_file':
        output_file = argslist[i+1]
    
    if argslist[i]=='--glove_embeddings_file':
        glove_file = argslist[i+1]
    
    if argslist[i]=='--vocabulary_input_file':
        vocab_path = argslist[i+1]
    i+=2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------- inputs ------------------
pre_tr = False #it will define a random embeddings of size vocab_size and then load the model

print_k = True
test_file = test_file
batch_size = 128
input_size = 100 + 50*char_level
dropout=0.5
# ----------------------------------

# ----------------- dicts load ---------------------
tag_idx = {'O':0, 'B-tim':1, 'I-tim':2, 'B-gpe':3, 'I-gpe':4, 'B-eve':5, 'I-eve':6, 'B-per':7, 'I-per':8, 'B-art':9, 'I-art':10, 'B-org':11, 'I-org':12, 'I-geo':13, 'B-geo':14, 'B-nat':15, 'I-nat':16}
tag_list = ['O', 'B-tim', 'I-tim', 'B-gpe', 'I-gpe', 'B-eve', 'I-eve', 'B-per', 'I-per', 'B-art', 'I-art', 'B-org', 'I-org', 'I-geo', 'B-geo', 'B-nat', 'I-nat']
num_tags = len(tag_list)
#indices
a_file = open(vocab_path,"rb")
vocab = pickle.load(a_file)
word_idx = vocab['word_idx']
char_idx = vocab['char_idx']
vocab_size = len(word_idx) 
char_size = len(char_idx)

torch.cuda.empty_cache()
if crf:
    use_hidden_layer = True
    model = BiLSTM_crf(pre_trained=None, input_size=input_size, dropout=dropout, use_hidden_layer = use_hidden_layer, vocab_size=vocab_size, char_size=char_size, pre_tr=pre_tr, norm=norm, char_level=char_level).to(device)
    model.load_state_dict(torch.load(model_path))
    test_model_crf(model, batch_size=batch_size, device=device, tag_list=tag_list,  output_path=output_file, test_file=test_file, word_idx=word_idx, char_idx=char_idx, tag_idx=tag_idx)
else:
    model = BiLSTM(pre_trained=None, input_size=input_size, num_tags=num_tags, dropout=dropout, pre_tr=pre_tr, norm=norm, char_level=char_level, char_size=char_size, vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load(model_path))
    test_model_output(model, batch_size=batch_size, device=device, tag_list=tag_list, output_path=output_file, test_file=test_file, word_idx=word_idx, char_idx=char_idx, tag_idx=tag_idx)