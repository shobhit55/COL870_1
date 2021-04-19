def glove_save(glove_file):

    data = pd.read_csv(glove_file, sep=" ", header=None, quoting=csv.QUOTE_NONE)
    tensor = torch.zeros(data.count()[0]+2, 100) #0th row represents 'no word'
    idx = {}
    idx['Nil']=0
    idx['Unk']=1
    for i in range(data.count()[0]):
        idx[data.iloc[i][0]] = i+2 #index 0 reserved for no word, 1 reserved for out of vocab word
        tensor[i+2,:] = torch.tensor(list(map(float, data.iloc[i][1:])))
    torch.save(tensor, "pre_trained.pt")
    a_file = open("word_idx_from_glove.pkl", "wb")
    pickle.dump(idx, a_file)
    a_file.close()

def train_vocab_save(train_file)
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
    a_file = open("word_idx_from_train.pkl", "wb")
    pickle.dump(train_data_token_dict, a_file)
    a_file.close()

def char_ind_save(train_file):
    #making char_idx
    data_file = open(train_file).read()
    chr_idx={}
    chr_idx['Nil']=0
    chr_idx['Unk']=1
    for i, char in enumerate(set(data_file)):
        chr_idx[char] = i+2
    a_file = open("char_idx_from_train.pkl", "wb")
    pickle.dump(chr_idx, a_file)
    a_file.close()