import torch
import torch.nn as nn
import torch.nn.functional as F

# processing bar
from tqdm import tqdm
import numpy as np

from cnn_hyperparams import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    def __init__(self, input_size, in_channels, kernel_size_row, 
                 kernel_size_col, num_classes):
        super(CNN, self).__init__()
        self.conv11 = nn.Conv2d(in_channels=1, out_channels=21, 
                               kernel_size=(15, 21),
                              stride = 1, padding=(7, 0))
        
#         self.conv12 = nn.Conv2d(in_channels=7, out_channels=1, 
#                                kernel_size=(3, 3),
#                               stride = 1, padding=(1, 1))
        
        self.conv5 = nn.Conv1d(in_channels=21, out_channels=1,
                              kernel_size=1, stride=1, 
                               padding=0)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = F.relu(self.conv11(x))
        
        x = self.conv5(x.squeeze(3))
        x = self.sigmoid(x)
        # x = (x - x.min())/(x.max()-x.min())
        return x
    
    def train_batch(self, train_data, optimizer, criterion, num_batch):
        # set training state to model
        self.to(device)
        self.train()
        n_samples = len(train_data)
        with tqdm(total=num_batch, position=0) as progress_bar:
            for i in range(num_batch):
                # get current batch dataset
                bat_data = train_data[i*batch_size:min((i+1)*batch_size, n_samples)]
                bat_loss = 0
                for batch_idx, (data, target) in enumerate(bat_data):
                    data = data.to(device)
                    target = target.to(device)
                    # forward
                    scores = self(data.reshape(1, 1, data.shape[0], data.shape[1]))
                    loss = criterion(scores.squeeze(0).squeeze(0), target)
                    # ERROR
                    bat_loss = bat_loss + loss
                # backword
                optimizer.zero_grad()
                bat_loss.backward()
                #gradient descent or adam step
                optimizer.step()
                progress_bar.update(1)
                
    def val_batch(self, val_data, criterion): 
        # set evaluation state to the model
        self.to(device)
        self.eval()
        losses = []
        
        class_probs = []
        class_label = []
        
        # no gradient needed
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_data):
                data = data.to(device)
                target = target.to(device)
                # forward
                scores = self(data.reshape(1, 1, data.shape[0], data.shape[1]))
                loss = criterion(scores.squeeze(0).squeeze(0), target)
                # ERROR
                losses.append(loss.cpu()) # loss for each batch
                # save for ploting curve
                class_probs.append(scores.squeeze(0).squeeze(0).cpu())
                class_label.append(target.cpu())
                
        # overall loss
        loss = np.mean(losses)
        return loss, class_probs, class_label
