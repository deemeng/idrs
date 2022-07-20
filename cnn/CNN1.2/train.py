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
from cnn_hyperparams import *
from model import CNN
from utils import load_dataset, get_num_class, rocPlot
# plots
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score

# 
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('CNN1.2board')


##
# Log Info
##
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
logging.info('Device: ' + device.type)


#####
# 1. Dataset
#####

# Load training and test dataset
logging.info('Loading training file ...')
train_data, train_plens = load_dataset(train_fpath)
logging.info('Loading test file ...')
test_data, test_plens = load_dataset(test_fpath)

# Number of training and test dataset
n_train = len(train_data)
n_test = len(test_data)
logging.info('Training set: ' + str(n_train))
logging.info('Test set:     ' + str(n_test))

train_c0, train_c1 = get_num_class(train_data)
test_c0, test_c1 = get_num_class(test_data)
logging.info(f'train c0, c1: {train_c0}, {train_c1}')
logging.info(f'test c0, c1: {test_c0}, {test_c1}')

#####
# 2. Initilization
#####

# init model
model = CNN(input_size, in_channels, kernel_size_row, 
                 kernel_size_col, num_classes)
# propogation of two classes
dis_sum = 0 
for batch_idx, (data, target) in enumerate(train_data):
    dis_sum = dis_sum + target.sum()
class_weights = torch.tensor([dis_sum/sum(train_plens), 1-dis_sum/sum(train_plens)], dtype=torch.float)
logging.info('class weights: ' + str(class_weights))

# Loss & Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#####
# 3. Training & Evaluation
#####
logging.warning(f"model_name: {model_name} \n \
                batch_size: {batch_size} \n \
                num_epochs: {num_epochs} \n \
                learning_rate: {learning_rate}")
'''
For each epoch:
    1. training
    2. calculate acc & loss for training dataset
    3. calculate acc & loss for val dataset
    4. add accs & losses to tensorboard
    5. finding the best val epoch
'''
# shuffled_data = p_data.copy()
num_batch = (n_train+batch_size-1)//batch_size
for epoch in range(num_epochs):
    logging.warning('*********************************')
    logging.warning(f'epoch: {epoch}')
    logging.warning('*********************************')
    
    print(f'epoch: {epoch}')
    ################
    # 1. Training
    ################
    
    plot_pr_roc = (epoch % plot_nEpoch == 0)
    # plot the last epoch
    if epoch==num_epochs-1:
        plot_pr_roc=True
        
    random.shuffle(train_data)
    # training
    model.train_batch(train_data, optimizer, criterion, num_batch)
    
    # acc & loss
    train_loss, t_class_probs, t_class_label = model.val_batch(train_data, criterion)
    val_loss, v_class_probs, v_class_label = model.val_batch(test_data, criterion)
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

        writer.add_pr_curve(f'TRAIN: pr_curve e{epoch}', train_label, train_probs, 0)

        val_probs = torch.cat([batch for batch in v_class_probs])
        val_label = torch.cat(v_class_label)

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