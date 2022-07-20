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
import torchvision.transforms as transforms
import random

# import hyperparameters
from brnn_hyperparams import *
from model import BRNN
from utils import load_dataset, get_num_class, rocPlot

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
logging.basicConfig(filename=log_name+'_test', 
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
logging.warning('Loading training file ...')
train_data, train_plens = load_dataset(train_fpath)
logging.warning('Loading test file ...')
test_data, test_plens = load_dataset(test_fpath)

# Number of training and test dataset
n_train = len(train_data)
n_test = len(test_data)
logging.warning('Training set: ' + str(n_train))
logging.warning('Test set:     ' + str(n_test))

train_c0, train_c1 = get_num_class(train_data)
test_c0, test_c1 = get_num_class(test_data)
logging.warning(f'train c0, c1: {train_c0}, {train_c1}')
logging.warning(f'test c0, c1: {test_c0}, {test_c1}')

#####
# 2. Initilization
#####

# init model
model = BRNN(input_size, hidden_size, num_layers, num_classes)
# propogation of two classes
dis_sum = 0 
for batch_idx, (data, target) in enumerate(train_data):
    dis_sum = dis_sum + target.sum()
class_weights = torch.tensor([dis_sum/sum(train_plens), 1-dis_sum/sum(train_plens)], dtype=torch.float)
logging.warning('class weights: ' + str(class_weights))

# Loss & Optimizer
# criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)


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
    writer.add_scalar("TRAIN: loss", train_loss, epoch)
    #writer.add_scalar("TRAIN: acc", train_acc, epoch)
    
    writer.add_scalar("VAL: loss", val_loss, epoch)
    #writer.add_scalar("VAL: acc", val_acc, epoch)
    
    ##
    # PR-curve & ROC-curve
    ##
    if plot_pr_roc:
        train_probs = torch.cat([batch for batch in t_class_probs])
        train_label = torch.cat(t_class_label)

        writer.add_pr_curve(f'TRAIN: pr_curve e{epoch}', train_label.reshape(-1), train_probs, 0)

        val_probs = torch.cat([batch for batch in v_class_probs])
        val_label = torch.cat(v_class_label)

        writer.add_pr_curve(f'VAL: pr_curve e{epoch}', val_label.reshape(-1), val_probs, 0)

        roc_fig = rocPlot(train_label.reshape(-1), train_probs, val_label.reshape(-1), val_probs)
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
