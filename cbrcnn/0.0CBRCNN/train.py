'''
This .py file includes 
1. init cnn model
2. choose Loss function and optimizing method
3. Train and evaluate model
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torchvision.transforms as transforms
import random

# import hyperparameters
from cbrcnn_hyperparams import *
from model import CBRCNN
from utils import ProteinDataset, get_num_class, rocPlot, pad_packed_collate

import numpy as np

# processing bar
from tqdm import tqdm

# plots
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score

# 
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter(f'{model_name}board')

import logging
logging.root.setLevel(logging.INFO)
# logging.basicConfig(level=logging.NOTSET)
logging.basicConfig(filename=log_name, 
                    filemode='a',
                    format='%(asctime)s %(message)s', 
                    datefmt='%m/%d/%Y %I:%M:%S %P',
                    level=logging.INFO)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.warning('Device: ' + device.type)
#####
# 1. Dataset
#####

# Load training and test dataset
# !!!!
# file need extra one empty line
# !!!!
logging.warning('Loading training file ...')
train_ds = ProteinDataset(train_fpath)
logging.warning('Loading test file ...')
test_ds = ProteinDataset(test_fpath)

logging.warning('Training set: ' + str(len(train_ds.p_lens)))
logging.warning('Test set:     ' + str(len(test_ds.p_lens)))

# dataloader
# using pad_packed_collate to deal with padding
train_dl = DataLoader(train_ds, batch_size = batch_size, num_workers = 72, shuffle=True, collate_fn=pad_packed_collate)
test_dl = DataLoader(test_ds, batch_size = batch_size, num_workers = 72, shuffle=False, collate_fn=pad_packed_collate)
      
#####
# 2. Initilization
#####

# init model
model = CBRCNN(input_size, hidden_size, num_layers, num_classes, dropP)
# propogation of two classes

# criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#####
# 3. Training & Evaluation
#####
logging.warning(f"model_name: {model_name} \n \
                batch_size: {batch_size} \n \
                num_epochs: {num_epochs} \n \
                learning_rate: {learning_rate}")

for epoch in range(num_epochs):
    logging.warning('*********************************')
    logging.warning('epoch: ' + str(epoch))
    logging.warning('*********************************')
    ################
    # 3.1. Training
    ################
    logging.warning('1. TRAINING:')
    # plot pr and roc curve or not
    plot_pr_roc = (epoch % plot_nEpoch == 0)

    # training
    model.train_batch(train_dl, optimizer, criterion)
    
    # acc & loss
    train_loss, t_class_probs, t_class_label = model.val_batch(train_dl, criterion)
    val_loss, v_class_probs, v_class_label = model.val_batch(test_dl, criterion)
    
    # val_accs.append(val_acc)
    
    print('Training: Loss:')
    print(train_loss)
    
    print('VAL: Loss:')
    print(val_loss)
    
    logging.warning('Training:')
    logging.warning(f'Loss: {train_loss}')

    logging.warning('Validation:')
    logging.warning(f'Loss: {val_loss}')
    
    ##
    # for tensorboard plots
    ##
    writer.add_scalars("TRAIN & VAL Loss", {'TRAIN': train_loss, 
                                           'VAL': val_loss}, epoch)
    
    train_probs = torch.cat([batch.reshape(-1) for batch in t_class_probs])
    train_label = torch.cat([lab.reshape(-1) for lab in t_class_label])
    
    val_probs = torch.cat([batch.reshape(-1) for batch in v_class_probs])
    val_label = torch.cat([lab.reshape(-1) for lab in v_class_label])
    
    writer.add_scalars("AUC Score", {'TRAIN': roc_auc_score(train_label, train_probs, average=None),
                                    'VAL': roc_auc_score(val_label, val_probs, average=None)}, epoch)
    
    ##
    # PR-curve & ROC-curve
    ##
    if plot_pr_roc:
        writer.add_pr_curve(f'TRAIN: pr_curve e{epoch}', train_label, train_probs, 0)
        writer.add_pr_curve(f'VAL: pr_curve e{epoch}', val_label, val_probs, 0)

        roc_fig = rocPlot(train_label, train_probs, val_label, val_probs)
        writer.add_figure(f'Train vs VAL: roc_curve e{epoch}', roc_fig)
    ##
    # Save model every m epochs
    ##
    if epoch % checkpoint_m == 0:
        cPATH = f"checkpoint/{model_name}_{epoch}.pth"
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                    }, cPATH)
        
logging.warning(f"model_name: {model_name} \n \
                batch_size: {batch_size} \n \
                 num_epochs: {num_epochs} \n \
                learning_rate: {learning_rate} \n ")
                # best_val_epoch: {int(np.argmax(val_accs)+1)}") # the best val epoch

writer.flush()
writer.close()

#####
# 4. Save Model
#####
torch.save(model.state_dict(), 'model.pth')
