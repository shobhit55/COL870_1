from dataset_ner import NERGmbData, NERGmbData_test
from models_ner import BiLSTM_crf, BiLSTM
from train_loops_ner import train_model_crf, train_model
from utils_ner import get_loaders



--initialization [random | glove ] --char_embeddings [ 0 | 1 ] --layer_normalization [ 0 | 1 ] --crf [ 0 | 1 ]
 --output_file <path to the trained model> --data_dir <directory containing data> 
 --glove_embeddings_file <path to file containing glove embeddings> --vocabulary_output_file <path to the file in which vocabulary will be written>

# ---------------- inputs ------------------
cont_train = False
model_folder = '' #'1.1_pre_tr_dropout'
save_folder = '' #'1.1_pre_tr_dropout_outputtest'

pre_tr = True #run data header also, if this changed
norm = False
crf = True
char_level = False
dropout = 0.5
epochs = 1

glove_file

batch_size = 128
no_fine_tune = False
input_size = 100 +50*char_level
patience = 4
lr = 0.001
optim_key = 'Adam' #'AdaDel' #'SGD', Adam for dropout
# ----------------------------------

# ----------------- dicts load ---------------------
tag_idx = {'O':0, 'B-tim':1, 'I-tim':2, 'B-gpe':3, 'I-gpe':4, 'B-eve':5, 'I-eve':6, 'B-per':7, 'I-per':8, 'B-art':9, 'I-art':10, 'B-org':11, 'I-org':12, 'I-geo':13, 'B-geo':14, 'B-nat':15, 'I-nat':16}
tag_list = ['O', 'B-tim', 'I-tim', 'B-gpe', 'I-gpe', 'B-eve', 'I-eve', 'B-per', 'I-per', 'B-art', 'I-art', 'B-org', 'I-org', 'I-geo', 'B-geo', 'B-nat', 'I-nat']
num_tags = len(tag_list)
#char indices
a_file = open("char_idx_from_train.pkl","rb")
char_idx = pickle.load(a_file)
# word_idx and pre_trained tensors
if pre_tr:
    pre_trained = torch.load("pre_trained.pt")
    a_file = open("word_idx_from_glove.pkl","rb")
    word_idx = pickle.load(a_file) #word indices
else:
    pre_trained = None
    a_file = open("word_idx_from_train.pkl","rb")
    word_idx = pickle.load(a_file) #word indices
vocab_size = len(word_idx) 
char_size = len(char_idx)
#------------------------------------------------------------------

torch.cuda.empty_cache()
train_loader, val_loader, test_loader = get_loaders(batch_size)
if crf:
    use_hidden_layer = True
    model = BiLSTM_crf(pre_trained=pre_trained, use_hidden_layer = use_hidden_layer, vocab_size=vocab_size, char_size=char_size).to(device)
    model = train_model_crf(model, train_loader, val_loader)
else:
    model = BiLSTM(pre_trained, input_size, num_tags, dropout=dropout, pre_tr=pre_tr, norm=norm, char_level=char_level).to(device)
    model, train_loss, valid_loss, train_ac, val_ac, f1_mi_train, f1_ma_train, f1_mi_val, f1_ma_val = train_model(model, train_loader, val_loader)
