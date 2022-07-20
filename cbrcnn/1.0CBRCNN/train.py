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
from model import CBRCNNStage1, CBRCNNStage2, train_batch, val_batch
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
modelStage1 = CBRCNNStage1(input_size, hidden_size, num_layers, num_classes, dropP)
modelStage2 = CBRCNNStage2(input_size_stage2, hidden_size_stage2, num_layers, num_classes, dropP)

# propogation of two classes

# criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()

optimizerStage1 = optim.Adam(modelStage1.parameters(), lr=learning_rate)
optimizerStage2 = optim.Adam(modelStage2.parameters(), lr=learning_rate)

#####
# 3. Training & Evaluation
#####
logging.warning(f"model_name: {model_name}_s1&s2 \n \
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
    train_batch(modelStage1, modelStage2, train_dl, optimizerStage1, optimizerStage2, criterion)
    
    # acc & loss
    train_loss_s1, train_loss_s2, t_class_probs_s1, t_class_probs_s2, t_class_label = val_batch(modelStage1, modelStage2, train_dl, criterion)
    val_loss_s1, val_loss_s2, v_class_probs_s1, v_class_probs_s2, v_class_label = val_batch(modelStage1, modelStage2, test_dl, criterion)
    
    # val_accs.append(val_acc)
    
    print('Training [Stage 1]: Loss:')
    print(train_loss_s1)
    
    print('Training [Stage 2]: Loss:')
    print(train_loss_s2)
    
    print('VAL [Stage 1]: Loss:')
    print(val_loss_s1)
    
    print('VAL [Stage 2]: Loss:')
    print(val_loss_s2)
    
    logging.warning('Training:')
    logging.warning(f'Loss s1, s2: {train_loss_s1}, {train_loss_s2}')

    logging.warning('Validation:')
    logging.warning(f'Loss s1, s2: {val_loss_s1}, {val_loss_s2}')
    
    ##
    # for tensorboard plots
    ##
    writer.add_scalars("TRAIN & VAL Loss", {'TRAIN S1': train_loss_s1,
                                            'TRAIN S2': train_loss_s2,
                                           'VAL S1': val_loss_s1,
                                           'VAL S2': val_loss_s2}, epoch)
    
    train_probs_s1 = torch.cat([batch.reshape(-1) for batch in t_class_probs_s1])
    train_probs_s2 = torch.cat([batch.reshape(-1) for batch in t_class_probs_s2])
    train_label = torch.cat([lab.reshape(-1) for lab in t_class_label])
    
    val_probs_s1 = torch.cat([batch.reshape(-1) for batch in v_class_probs_s1])
    val_probs_s2 = torch.cat([batch.reshape(-1) for batch in v_class_probs_s2])
    val_label = torch.cat([lab.reshape(-1) for lab in v_class_label])
    
    writer.add_scalars("AUC Score", {'TRAIN S1': roc_auc_score(train_label, train_probs_s1, average=None),
                                     'TRAIN S2': roc_auc_score(train_label, train_probs_s2, average=None),
                                    'VAL S1': roc_auc_score(val_label, val_probs_s1, average=None),
                                    'VAL S2': roc_auc_score(val_label, val_probs_s2, average=None)}, epoch)
    
    ##
    # PR-curve & ROC-curve
    ##
    
    if plot_pr_roc:
        writer.add_pr_curve(f'TRAIN S1: pr_curve e{epoch}', train_label, train_probs_s1, 0)
        writer.add_pr_curve(f'VAL S1: pr_curve e{epoch}', val_label, val_probs_s1, 0)
        
        writer.add_pr_curve(f'TRAIN S2: pr_curve e{epoch}', train_label, train_probs_s2, 0)
        writer.add_pr_curve(f'VAL S2: pr_curve e{epoch}', val_label, val_probs_s2, 0)

        roc_fig_s1 = rocPlot(train_label, train_probs_s1, val_label, val_probs_s1)
        roc_fig_s2 = rocPlot(train_label, train_probs_s2, val_label, val_probs_s2)
        writer.add_figure(f'[Stage 1] Train vs VAL: roc_curve e{epoch}', roc_fig_s1)
        writer.add_figure(f'[Stage 2] Train vs VAL: roc_curve e{epoch}', roc_fig_s2)
    
    ##
    # Save model every m epochs
    ##
    if epoch % checkpoint_m == 0:
        # Stage 1 model
        cPATH_s1 = f"checkpoint/{model_name}_s1_{epoch}.pth"
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': modelStage1.state_dict(),
                    'optimizer_state_dict': optimizerStage1.state_dict(),
                    'loss': train_loss_s1,
                    }, cPATH_s1)
        
        # Stage 2 model
        cPATH_s2 = f"checkpoint/{model_name}_s2_{epoch}.pth"
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': modelStage2.state_dict(),
                    'optimizer_state_dict': optimizerStage2.state_dict(),
                    'loss': train_loss_s2,
                    }, cPATH_s2)
        
logging.warning(f"model_name: {model_name}_s1&S2 \n \
                batch_size: {batch_size} \n \
                 num_epochs: {num_epochs} \n \
                learning_rate: {learning_rate} \n ")
                # best_val_epoch: {int(np.argmax(val_accs)+1)}") # the best val epoch

writer.flush()
writer.close()

#####
# 4. Save Model
#####
torch.save(modelStage1.state_dict(), 'model_s1.pth')
torch.save(modelStage2.state_dict(), 'model_s2.pth')
