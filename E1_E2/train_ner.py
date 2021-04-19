--initialization [random | glove ] --char_embeddings [ 0 | 1 ] --layer_normalization [ 0 | 1 ] --crf [ 0 | 1 ]
 --output_file <path to the trained model> --data_dir <directory containing data> 
 --glove_embeddings_file <path to file containing glove embeddings> --vocabulary_output_file <path to the file in which vocabulary will be written>
glove_file
pre_tr

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

