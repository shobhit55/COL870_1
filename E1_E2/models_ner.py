from torch import jit
from torch import nn
import torch

class LayerNormLSTMCell(jit.ScriptModule): #jit.ScriptModule, nn.Module
    def __init__(self, input_size, hidden_size, decompose_layernorm=False):
        super(LayerNormLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.linear_ih = nn.Linear(input_size, 4 * hidden_size) #1
        self.linear_hh = nn.Linear(hidden_size, 4 * hidden_size) #1
        
        ln = nn.LayerNorm
        self.layernorm_i = ln(4 * hidden_size, elementwise_affine=False)
        self.layernorm_h = ln(4 * hidden_size, elementwise_affine=False)
        self.layernorm_c = ln(hidden_size, elementwise_affine=False)
    
    @jit.script_method
    def forward(self, input, state):
    # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]
        hx, cx = state
        igates = self.layernorm_i(self.linear_ih(input))
        hgates = self.layernorm_h(self.linear_hh(hx))

        gates = igates + hgates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        cy = self.layernorm_c((forgetgate * cx) + (ingate * cellgate))
        hy = outgate * torch.tanh(cy)
        return hy, cy

class BiLSTM_emb(nn.Module):
    def __init__(self, input_size, char_size):
        super(BiLSTM_emb, self).__init__()
        self.input_size = input_size
        self.embedding = nn.Embedding(char_size, input_size)
        self.LSTMlayer = nn.LSTM(input_size, input_size, batch_first=True, bidirectional=True) #forward, 25d word embeddings and 25d hidden state

    def forward(self, x): #x - shape(batch_size, max_word_len), int indices
        x = self.embedding(x)
        
        _, (h_n, c_n) = self.LSTMlayer(x)
        return h_n.view(-1, 2*self.input_size) #batch_size, 2*hidden_dim

class BiLSTM(nn.Module):
    def __init__(self, pre_trained, input_size, num_tags, char_size, vocab_size, dropout, pre_tr, norm, char_level):
        super(BiLSTM, self).__init__()
        if pre_tr:
            self.embedding = nn.Embedding.from_pretrained(pre_trained, freeze=False) #Fine tuning allowed
        else:
            self.embedding = nn.Embedding(vocab_size, input_size-char_level*50)
        
        if char_level:
            self.char_embed = BiLSTM_emb(input_size = 25, char_size=char_size) #input - batch_size, max_batch_len_words

        self.dropout = nn.Dropout(p=dropout)
        if norm:
            self.LSTMcelle = LayerNormLSTMCell(input_size, input_size) #forward, 100 or 150d word embeddings and 100 or 150d hidden state
            self.LSTMcellr = LayerNormLSTMCell(input_size, input_size) #reverse, 100 or 150d word embeddings and 100 or 150d hidden state
        else:
            self.LSTMlayer = nn.LSTM(input_size, input_size, batch_first=True, bidirectional=True) #100d word embeddings and 100 hidden state

        self.linear = nn.Linear(input_size*2, num_tags)
        
        self.norm = norm
        self.char_level = char_level

    def forward(self, x): #x - shape(batch_size, max_len of sentences, max_word_len+1), int indices
        word_embT = self.embedding(x[:,:,0])
        batch_size, max_len = x.shape[0], x.shape[1]
        
        if self.char_level:
            char_embT = torch.zeros(batch_size, max_len, 50, device=x.device)
            for i in range(max_len):
                char_embT[:,i,:] = self.char_embed(x[:,i,1:])
            x = torch.cat((word_embT,char_embT), dim=2)
        else:
            x = word_embT.clone()
        
        input_size = x.shape[2]

        x = self.dropout(x)

        if self.norm:
            xe = torch.zeros(batch_size, max_len, input_size, device=x.device) #stores forward hidden states
            xr = torch.zeros(batch_size, max_len, input_size, device=x.device) #stores reverse hidden states
            xrT_t = torch.zeros(batch_size, input_size, device=x.device)
            crT_t = torch.zeros(batch_size, input_size, device=x.device)
            xet = torch.zeros(batch_size, input_size, device=x.device)
            cet = torch.zeros(batch_size, input_size, device=x.device)
            
            # pdb.set_trace()
            for t in range(max_len):
                xet, cet = self.LSTMcelle(x[:,t,:],(xet,cet))
                xe[:,t,:] = xet.clone()
                xrT_t, crT_t = self.LSTMcellr(x[:,max_len-1-t,:], (xrT_t,crT_t))
                xr[:,max_len-1-t,:] = xrT_t.clone()
            
            
            xrT_t.detach()
            crT_t.detach()
            xet.detach()
            cet.detach()

            x = torch.cat((xe,xr), dim=2)
        else:
            x, _ = self.LSTMlayer(x)
        
        x = nn.functional.log_softmax(self.linear(x), dim=2)
        return x

    def init_weights(self):
      for m in self.modules():
        if isinstance(m, nn.LSTM):
            nn.init.kaiming_uniform_(m.weight)
        if isinstance(m, nn.LSTMCell):
            nn.init.kaiming_uniform_(m.weight)
        if isinstance(m, nn.Linear):
          nn.init.kaiming_uniform_(m.weight)
          nn.init.constant_(m.bias,0.001)

class CRF(nn.Module):
    def __init__(
        self, nb_labels, bos_tag_id, eos_tag_id, batch_first=True
    ):
        super().__init__()
        self.nb_labels = nb_labels
        self.BOS_TAG_ID = bos_tag_id
        self.EOS_TAG_ID = eos_tag_id
        self.batch_first = batch_first
        self.transitions = nn.Parameter(torch.empty(self.nb_labels, self.nb_labels))
        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        self.transitions.data[:, self.BOS_TAG_ID] = -10000.0
        self.transitions.data[self.EOS_TAG_ID, :] = -10000.0

    def forward(self, emissions, tags, mask=None):
      nll = -self.log_likelihood(emissions, tags, mask=mask)
      return nll

    def log_likelihood(self, emissions, tags, mask=None):
        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.float)

        scores = self._compute_scores(emissions, tags, mask=mask)
        partition = self._compute_log_partition(emissions, mask=mask)
        return torch.mean(scores - partition)

    def _compute_scores(self, emissions, tags, mask):
      batch_size, seq_length = tags.shape
      scores = torch.zeros(batch_size, device=emissions.device) #.to(device)
      first_tags = tags[:, 0]
      last_valid_idx = mask.int().sum(1) - 1
      last_tags = tags.gather(1, last_valid_idx.unsqueeze(1)).squeeze()

      t_scores = self.transitions[self.BOS_TAG_ID, first_tags]
 
      e_scores = emissions[:, 0].gather(1, first_tags.unsqueeze(1).to(emissions.device)).squeeze()
      scores += e_scores + t_scores
      for i in range(1, seq_length):
          is_valid = mask[:, i]
          previous_tags = tags[:, i - 1]
          current_tags = tags[:, i]
          previous_tags = previous_tags*is_valid
          current_tags = current_tags*is_valid
          e_scores = emissions[:, i].gather(1, current_tags.unsqueeze(1)).squeeze()
          
          previous_tags_cpu = previous_tags
          current_tags_cpu = current_tags
          t_scores = self.transitions[previous_tags_cpu, current_tags_cpu]
          e_scores = e_scores * is_valid
          t_scores = t_scores * is_valid
          scores += e_scores + t_scores

      scores += self.transitions[last_tags, self.EOS_TAG_ID]
      return scores

    def _compute_log_partition(self, emissions, mask):
      batch_size, seq_length, nb_labels = emissions.shape

      # in the first iteration, BOS will have all the scores
      alphas = self.transitions[self.BOS_TAG_ID, :].unsqueeze(0) + emissions[:, 0]

      for i in range(1, seq_length):
          alpha_t = []

          for tag in range(nb_labels):
              e_scores = emissions[:, i, tag]
              e_scores = e_scores.unsqueeze(1)
              t_scores = self.transitions[:, tag]
              t_scores = t_scores.unsqueeze(0)
              scores = e_scores + t_scores + alphas
              alpha_t.append(torch.logsumexp(scores, dim=1))

          new_alphas = torch.stack(alpha_t).t()
          is_valid = mask[:, i].unsqueeze(-1)
          alphas = is_valid * new_alphas + (1 - is_valid) * alphas

      last_transition = self.transitions[:, self.EOS_TAG_ID]
      end_scores = alphas + last_transition.unsqueeze(0)
      return torch.logsumexp(end_scores, dim=1)

    def decode(self, emissions, mask=None):
      if mask is None:
          mask = torch.ones(emissions.shape[:2], dtype=torch.float)

      scores, sequences = self._viterbi_decode(emissions, mask)
      return scores, sequences

    def _viterbi_decode(self, emissions, mask):
        batch_size, seq_length, nb_labels = emissions.shape

        alphas = self.transitions[self.BOS_TAG_ID, :].unsqueeze(0) + emissions[:, 0]

        backpointers = []

        for i in range(1, seq_length):
            alpha_t = []
            backpointers_t = []

            for tag in range(nb_labels):
                e_scores = emissions[:, i, tag]
                e_scores = e_scores.unsqueeze(1)
                t_scores = self.transitions[:, tag]
                t_scores = t_scores.unsqueeze(0)
                scores = e_scores + t_scores + alphas
                max_score, max_score_tag = torch.max(scores, dim=-1)
                alpha_t.append(max_score)
                backpointers_t.append(max_score_tag)
            
            new_alphas = torch.stack(alpha_t).t()
            is_valid = mask[:, i].unsqueeze(-1).int()
            alphas = is_valid * new_alphas + (1 - is_valid) * alphas
            backpointers.append(backpointers_t)

        last_transition = self.transitions[:, self.EOS_TAG_ID]
        end_scores = alphas + last_transition.unsqueeze(0)
        max_final_scores, max_final_tags = torch.max(end_scores, dim=1)
        best_sequences = []
        emission_lengths = mask.int().sum(dim=1)
        
        for i in range(batch_size):
            sample_length = emission_lengths[i].item()
            sample_final_tag = max_final_tags[i].item()
            sample_backpointers = backpointers[: sample_length - 1]
            sample_path = self._find_best_path(i, sample_final_tag, sample_backpointers)
            best_sequences.append(sample_path)

        return max_final_scores, best_sequences

    def _find_best_path(self, sample_id, best_tag, backpointers):
        best_path = [best_tag]
        for backpointers_t in reversed(backpointers):
            best_tag = backpointers_t[best_tag][sample_id].item()
            best_path.insert(0, best_tag)

        return best_path

class BiLSTM_crf(nn.Module):
    def __init__(self, pre_trained, vocab_size, char_size, nb_labels = 17+2, use_hidden_layer= True, dropout = 0.5, pre_tr = True, norm = False, char_level = False, input_size = 100, num_tags=17):
        super(BiLSTM_crf, self).__init__()
        if pre_tr:
            self.embedding = nn.Embedding.from_pretrained(pre_trained, freeze=False) #Fine tuning allowed
        else:
            self.embedding = nn.Embedding(vocab_size, input_size-char_level*50)
        
        if char_level:
            self.char_embed = BiLSTM_emb(input_size = 25, char_size = char_size) #input - batch_size, max_batch_len_words

        self.dropout = nn.Dropout(p=dropout)
        if norm:
            self.LSTMcelle = LayerNormLSTMCell(input_size, input_size) #forward, 100 or 150d word embeddings and 100 or 150d hidden state
            self.LSTMcellr = LayerNormLSTMCell(input_size, input_size) #reverse, 100 or 150d word embeddings and 100 or 150d hidden state
        else:
            self.LSTMlayer = nn.LSTM(input_size, input_size, batch_first=True, bidirectional=True) #100d word embeddings and 100 hidden state

        if use_hidden_layer:
          self.linear = nn.Sequential( nn.Linear(2*input_size, 100), 
                                           nn.Linear(100, nb_labels))
        else:
          self.linear = nn.Linear(input_size*2, nb_labels)
                
        self.crf_layer = CRF(nb_labels, 17, 18, batch_first = True) #.to(device)
        self.norm = norm
        self.char_level = char_level

    def forward(self, x, target = None, test = False): #x - shape(batch_size, max_len of sentences, max_word_len+1), int indices # if test is True crf_layer.decode called 
        word_embT = self.embedding(x[:,:,0])
        # pdb.set_trace()
        batch_size, max_len = x.shape[0], x.shape[1]
        
        if self.char_level:
            char_embT = torch.zeros(batch_size, max_len, 50, device=x.device)
            for i in range(max_len):
                char_embT[:,i,:] = self.char_embed(x[:,i,1:])
            x = torch.cat((word_embT,char_embT), dim=2)
        else:
            x = word_embT.clone()
        
        input_size = x.shape[2]
        # pdb.set_trace()
        x = self.dropout(x)
        # pdb.set_trace()

        if self.norm:
            xe = torch.zeros(batch_size, max_len, input_size, device=x.device) #stores forward hidden states
            xr = torch.zeros(batch_size, max_len, input_size, device=x.device) #stores reverse hidden states
            xrT_t = torch.zeros(batch_size, input_size, device=x.device)
            crT_t = torch.zeros(batch_size, input_size, device=x.device)
            xet = torch.zeros(batch_size, input_size, device=x.device)
            cet = torch.zeros(batch_size, input_size, device=x.device)
            
            # pdb.set_trace()
            for t in range(max_len):
                xet, cet = self.LSTMcelle(x[:,t,:],(xet,cet))
                xe[:,t,:] = xet.clone()
                xrT_t, crT_t = self.LSTMcellr(x[:,max_len-1-t,:], (xrT_t,crT_t))
                xr[:,max_len-1-t,:] = xrT_t.clone()
            
            
            xrT_t.detach()
            crT_t.detach()
            xet.detach()
            cet.detach()

            x = torch.cat((xe,xr), dim=2)
        else:
            x, _ = self.LSTMlayer(x)
        
        x = self.linear(x)
        mask = (target != -1)
        mask = mask.int().to(x.device)

        if test:
          return self.crf_layer.decode(x, mask = mask)
        else:
          loss = self.crf_layer(x, target, mask = mask).to(x.device)
        return x, loss

    def init_weights(self):
      for m in self.modules():
        if isinstance(m, nn.LSTM):
            nn.init.kaiming_uniform_(m.weight)
        if isinstance(m, nn.LSTMCell):
            nn.init.kaiming_uniform_(m.weight)
        if isinstance(m, nn.Linear):
          nn.init.kaiming_uniform_(m.weight)
          nn.init.constant_(m.bias,0.001)