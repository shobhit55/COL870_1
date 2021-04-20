from torch.optim.lr_scheduler import StepLR
import torch
from torch import nn
import numpy as np
from utils_ner import EarlyStopping, criterion, accuracy, get_loader_test
# from seqeval.metrics import accuracy_score
# from seqeval.metrics import f1_score as seq_f1_score
# from seqeval.scheme import IOB2

def train_model(model, train_loader, val_loader, device, path, lr, print_k=False, optim_key='Adam', patience=7, epochs=75):
    print("Training Started...")
    #pr = f"char_level = {char_level} | norm = {norm} | pre_tr = {pre_tr} | dropout = {dropout} | r = {num_tags} | epochs = {epochs} | lr = {lr} | optimizer = {optim_key} | batch_size = {batch_size}"
    #save_print(pr, logger)
    optimizer_dict = { "AdaDel": torch.optim.Adadelta(model.parameters()), "Adam": torch.optim.Adam(model.parameters(), lr = lr), "SGD": torch.optim.SGD(model.parameters(), lr = lr) }
    
    # optim key for some reason didnt work earlier, make the optimizer directly using torch.optim.

    optimizer = optimizer_dict[optim_key] #torch.optim.SGD(model.parameters(), lr = lr)
    scheduler = StepLR(optimizer, step_size=25, gamma=0.5)
    if device==torch.device('cuda'):
        torch.cuda.synchronize()
    # train_ac = []
    # val_ac = []
    # f1_mi_train = []
    # f1_ma_train = []
    # f1_mi_val = []
    # f1_ma_val = []
    # train_losses = []
    valid_losses = []
    # avg_train_losses = []
    # avg_valid_losses = [] 
    start_epoch = 1
    #checkpt_folder = FOLDER + '/model_checkpoint.pt'
    early_stopping = EarlyStopping(path=path, patience=patience)
    for epoch in range(start_epoch, epochs+1):
    #   since = time.time()
    #   correct = 0.0
    #   total = 0
    #   train_acc = 0
    #   targ = torch.empty(0, device=device)
    #   out = torch.empty(0, device=device)
      model.train()

      for i, (input, target) in enumerate(train_loader):
          input, target = input.to(device), target.to(device)
          optimizer.zero_grad()

          output = model(input) #.to(device)
          loss = criterion(output, target)
          if print_k:
              print(i, loss)
        #   targ1 = target.view(-1)
        #   out1 = output.view(-1,output.shape[2])
        #   out = torch.cat((out, torch.argmax(out1, dim=1)), dim=0) #.to(device)
        #   targ = torch.cat((targ, targ1), dim=0)
        #   a, b = accuracy(output, target, 'all_O')
        #   correct+=a
        #   total+=b

          loss.backward() #scaler.scale(loss).backward() #
          nn.utils.clip_grad_norm_(model.parameters(), 5)
          optimizer.step() #scaler.step(optimizer) #
          #scaler.update() #1
        #   train_losses.append(loss.item())
      
    #   train_acc = correct/total # train data accuracy during this epoch
    #   train_ac.append(train_acc)
    #   f1_micro_train = f1_score(targ.cpu().detach(), out.cpu().detach(), labels=np.linspace(1,16, 16, dtype=int).tolist(), average='micro')
    #   f1_macro_train = f1_score(targ.cpu().detach(), out.cpu().detach(), labels=np.linspace(1,16, 16, dtype=int).tolist(), average='macro')
    #   f1_mi_train.append(f1_micro_train)
    #   f1_ma_train.append(f1_macro_train)

    #   correct = 0.0
    #   total = 0
    #   val_acc = 0
    #   targ = torch.empty(0, device=device)
    #   out = torch.empty(0, device=device)
      model.eval()
      for input, target in val_loader:
          input, target = input.to(device), target.to(device)
          output = model(input) #.to(device)
          loss = criterion(output, target)
          if print_k:
              print(loss)
        #   targ1 = target.view(-1)
        #   out1 = output.view(-1,output.shape[2])
        #   out = torch.cat((out, torch.argmax(out1, dim=1)), dim=0) #.to(device)
        #   targ = torch.cat((targ, targ1), dim=0) #.to(device)

        #   a, b = accuracy(output, target, 'all_O')
        #   correct+=a
        #   total+=b
          valid_losses.append(loss.item())
    #   val_acc = correct/total # val_data accuracy after this epoch
    #   val_ac.append(val_acc)
    #   f1_micro_val = f1_score(targ.cpu().detach(), out.cpu().detach(), labels=np.linspace(1,16, 16, dtype=int).tolist(), average='micro')
    #   f1_macro_val = f1_score(targ.cpu().detach(), out.cpu().detach(), labels=np.linspace(1,16, 16, dtype=int).tolist(), average='macro')
    #   f1_mi_val.append(f1_micro_val)
    #   f1_ma_val.append(f1_macro_val)

    #   train_loss = np.average(train_losses)
      valid_loss = np.average(valid_losses)
    #   avg_train_losses.append(train_loss)
    #   avg_valid_losses.append(valid_loss)
    #   epoch_len = len(str(epochs))

    #   save_print("-----------------------------------------------------------", logger)
    #   p_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
    #                 f'train_loss: {train_loss:.5f} ' +
    #                 f'valid_loss: {valid_loss:.5f} \n' +
    #                 f'train_acc: {train_acc:.2f} '+
    #                 f'val_acc: {val_acc:.2f} ' +
    #                 f'f1_micro_train: {f1_micro_train:.2f} '+
    #                 f'f1_macro_train: {f1_macro_train:.2f} '+
    #                 f'f1_micro_val: {f1_micro_val:.2f} '+
    #                 f'f1_macro_val: {f1_macro_val:.2f} ')
      
    #   save_print(p_msg, logger)
    #   train_losses = []
    #   valid_losses = []
    #   total_norm = 0
    #   for p in model.parameters():
    #     param_norm = p.grad.data.norm(2)
    #     total_norm += param_norm.item() ** 2
    #   total_norm = total_norm ** (1. / 2)
    #   print("total_gradient_norm:", total_norm)
      early_stopping(valid_loss, model)
      if early_stopping.early_stop:
          print("Early stopping")
          break
    #   time_elapsed = time.time() - since
    #   print('Epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    #   print()
      scheduler.step()

    model.load_state_dict(torch.load(path))
    print("Training Done...")
    return  model #, avg_train_losses, avg_valid_losses, train_ac, val_ac, f1_mi_train, f1_ma_train, f1_mi_val, f1_ma_val

def test_model_output(model, batch_size, device, tag_list, output_path, test_file, word_idx, tag_idx, char_idx, dat_kind='Test'): #saves the final metrics for input data 
    test_loader = get_loader_test(batch_size, test_file, word_idx=word_idx, char_idx=char_idx, tag_idx=tag_idx)
    # pred_tags = []
    # target_tags = []
    model.eval()
    data_file = open(test_file).read().split('\n\n')[:-1]

    for input, target, lengths, inds in test_loader: # lengths - batch
        input, target = input.to(device), target.to(device)
        output = model(input).to(device) #batch, max_len, max_word_len+1 -> batch, max_len, num_tags
        output = torch.argmax(output, dim=2) #batch, max_len, num_tags -> batch, max_len - output tag indices

        for j in range(len(inds)): #over sentences, output.shape[0]
            # pred_tags.append([])
            # target_tags.append([])
            sentj = data_file[inds[j]].split('\n')
            for k in range(lengths[j]): #over length of jth sentence
                wordi = sentj[k].split(' ')
                if len(wordi)!=4:
                    continue
                wordi[3] = tag_list[output[j][k]] # replace with prediction of kth word of jth sentence
                sentj[k] = ' '.join(wordi) #string
                # if target[j,k]!=tag_idx['O']:
                #     target_tags[-1].append(tag_list[target[j,k]])
                #     pred_tags[-1].append(tag_list[output[j,k]])
            data_file[inds[j]] = '\n'.join(sentj)

    data_file = '\n\n'.join(data_file)
    text_file = open(output_path, "w")
    text_file.write(data_file)
    text_file.close()
    # f1_micro = seq_f1_score(target_tags, pred_tags, scheme=IOB2, average='micro')
    # f1_macro = seq_f1_score(target_tags, pred_tags, scheme=IOB2, average='macro')
    # accuracy = accuracy_score(target_tags, pred_tags)

    # save_print(f'{dat_kind} Accuracy: {accuracy}, f1_micro: {f1_micro}, f1_macro: {f1_macro}', logger)

def train_model_crf(model, train_loader, val_loader, device, path, lr, print_k=False, optim_key='Adam', patience=7, epochs=75):
    print("Training Started...on...",device)
    # pr = f"char_level = {char_level} | norm = {norm} | pre_tr = {pre_tr} | dropout = {dropout} | r = {num_tags} | epochs = {epochs} | lr = {lr} | optimizer = {optim_key} | batch_size = {batch_size}"
    # save_print(pr, logger)
    optimizer_dict = { "AdaDel": torch.optim.Adadelta(model.parameters()), "Adam": torch.optim.Adam(model.parameters(), lr = lr), "SGD": torch.optim.SGD(model.parameters(), lr = lr) }
    optimizer = optimizer_dict[optim_key] #torch.optim.SGD(model.parameters(), lr = lr)
    scheduler = StepLR(optimizer, step_size=25, gamma=0.5)
    if device==torch.device('cuda'):
        torch.cuda.synchronize()
    # train_ac = []
    # val_ac = []
    # f1_mi_train = []
    # f1_ma_train = []
    # f1_mi_val = []
    # f1_ma_val = []
    # train_losses = []
    valid_losses = []
    # avg_train_losses = []
    # avg_valid_losses = [] 
    start_epoch = 1
    # checkpt_folder = FOLDER + '/model_checkpoint.pt'
    early_stopping = EarlyStopping(patience=patience, path = path)

    for epoch in range(start_epoch, epochs+1):
    #   since = time.time()
    #   correct = 0.0
    #   total = 0
    #   train_acc = 0
    #   targ = torch.empty(0, device=device)
    #   out = torch.empty(0, device=device)
      model.train()
      for i, (input, target) in enumerate(train_loader):
          input, target = input.to(device), target.to(device)
          optimizer.zero_grad()
          target = target.long()
          output, loss = model(input, target) #.to(device)
          if print_k:
              print(i, loss)
        #   print("---------------------------------")
        #   print(i, loss)
          # loss = criterion(output, target)
        #   targ1 = target.view(-1)
        #   out1 = output.view(-1,output.shape[2])
        #   out = torch.cat((out, torch.argmax(out1, dim=1)), dim=0) #.to(device)
        #   targ = torch.cat((targ, targ1), dim=0)
        #   a, b = accuracy(output, target, 'all_O')
        #   correct+=a
        #   total+=b
          loss.backward() #scaler.scale(loss).backward() #
          nn.utils.clip_grad_norm_(model.parameters(), 5)
          optimizer.step() #scaler.step(optimizer) #
          #scaler.update() #1
        #   train_losses.append(loss.item())

    #   train_acc = correct/total # train data accuracy during this epoch
    #   train_ac.append(train_acc)
      # f1_micro_train = f1_score(targ.cpu().detach(), out.cpu().detach(), labels=np.linspace(1,16, 16, dtype=int).tolist(), average='micro')
      # f1_macro_train = f1_score(targ.cpu().detach(), out.cpu().detach(), labels=np.linspace(1,16, 16, dtype=int).tolist(), average='macro')
      # f1_mi_train.append(f1_micro_train)
      # f1_ma_train.append(f1_macro_train)

      # correct = 0.0
      # total = 0
      # val_acc = 0
      # targ = torch.empty(0, device=device)
      # out = torch.empty(0, device=device)

      model.eval()
      for input, target in val_loader:
          input, target = input.to(device), target.to(device)
          target = target.long()
          output, loss = model(input, target) #.to(device)
          if print_k:
              print(loss)
  
        #   targ1 = target.view(-1)
        #   out1 = output.view(-1,output.shape[2])
        #   out = torch.cat((out, torch.argmax(out1, dim=1)), dim=0) #.to(device)
        #   targ = torch.cat((targ, targ1), dim=0) #.to(device)

      #     a, b = accuracy(output, target, 'all_O')
      #     correct+=a
      #     total+=b
          valid_losses.append(loss.item())
      # val_acc = correct/total # val_data accuracy after this epoch
      # val_ac.append(val_acc)
      # f1_micro_val = f1_score(targ.cpu().detach(), out.cpu().detach(), labels=np.linspace(1,16, 16, dtype=int).tolist(), average='micro')
      # f1_macro_val = f1_score(targ.cpu().detach(), out.cpu().detach(), labels=np.linspace(1,16, 16, dtype=int).tolist(), average='macro')
      # f1_mi_val.append(f1_micro_val)
      # f1_ma_val.append(f1_macro_val)

    #   train_loss = np.average(train_losses)
      valid_loss = np.average(valid_losses)
    #   avg_train_losses.append(train_loss)
      # avg_valid_losses.append(valid_loss)
      # epoch_len = len(str(epochs))

      # save_print("-----------------------------------------------------------", logger)
      # p_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
      #               f'train_loss: {train_loss:.5f} ' +
      #               f'valid_loss: {valid_loss:.5f} \n' +
      #               f'train_acc: {train_acc:.2f} '+
      #               f'val_acc: {val_acc:.2f} ' +
      #               f'f1_micro_train: {f1_micro_train:.2f} '+
      #               f'f1_macro_train: {f1_macro_train:.2f} '+
      #               f'f1_micro_val: {f1_micro_val:.2f} '+
      #               f'f1_macro_val: {f1_macro_val:.2f} ')
      
      # save_print(p_msg, logger)
      # train_losses = []
      # valid_losses = []
      # total_norm = 0
      # for p in model.parameters():
      #   param_norm = p.grad.data.norm(2)
      #   total_norm += param_norm.item() ** 2
      # total_norm = total_norm ** (1. / 2)
      # print("total_gradient_norm:", total_norm)
      early_stopping(valid_loss, model)
      if early_stopping.early_stop:
          print("Early stopping")
          break
    #   time_elapsed = time.time() - since
    #   print('Epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    #   print()
      scheduler.step()

    model.load_state_dict(torch.load(path))
    print("Training Done...")
    # return  model, avg_train_losses, avg_valid_losses, train_ac, val_ac, f1_mi_train, f1_ma_train, f1_mi_val, f1_ma_val
    return  model#, avg_train_losses, train_ac, f1_mi_train, f1_ma_train

def test_model_crf(model, batch_size, device, tag_list, output_path, test_file, word_idx, tag_idx, char_idx, dat_kind='Test'): #saves the final metrics for input data 
    test_loader = get_loader_test(batch_size, test_file, word_idx=word_idx, tag_idx=tag_idx, char_idx=char_idx)
    # pred_tags = []
    # target_tags = []
    data_file = open(test_file).read().split('\n\n')[:-1]

    model.eval()
    for input, target, lengths, inds in test_loader:
        input, target = input.to(device), target.to(device)
        scores, output = model(input, target, test = True)

        for j in range(len(inds)): #over sentences
            # pred_tags.append([])
            # target_tags.append([])
            sentj = data_file[inds[j]].split('\n')
            for k in range(lengths[j]):
                wordi = sentj[k].split(' ')
                if len(wordi)!=4:
                    continue
                wordi[3] = tag_list[output[j][k]] # replace with prediction of kth word of jth sentence
                sentj[k] = ' '.join(wordi) #string
                # if target[j,k]!=tag_idx['O']:
                #     target_tags[-1].append(tag_list[target[j,k]])
                #     pred_tags[-1].append(tag_list[output[j][k]])
            data_file[inds[j]] = '\n'.join(sentj)

    data_file = '\n\n'.join(data_file)
    text_file = open(output_path, "w")
    text_file.write(data_file)
    text_file.close()
    # f1_micro = seq_f1_score(target_tags, pred_tags, scheme=IOB2, average='micro')
    # f1_macro = seq_f1_score(target_tags, pred_tags, scheme=IOB2, average='macro')
    # accuracy = accuracy_score(target_tags, pred_tags)

    # save_print(f'{dat_kind} Accuracy: {accuracy}, f1_micro: {f1_micro}, f1_macro: {f1_macro}', logger)