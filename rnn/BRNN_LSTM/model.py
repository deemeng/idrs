import torch
import torch.nn as nn
import torch.nn.functional as F
# processing bar
from tqdm import tqdm
from brnn_hyperparams import *
import numpy as np

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# set device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                         bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        # Forward Prop
        # out, (hn, cn) = self.lstm(x,  (h0, c0))
        out, _ = self.lstm(x,  (h0, c0))
        # all training example, last hidden state, all 
        out = self.fc(out[:,-1, :])
        out = self.sigmoid(out)
        return out
    
    def train_batch(self, train_loader, optimizer, criterion):
        # set training state to model
        self.to(device)
        self.train()
        
        with tqdm(total=len(train_loader), position=0) as progress_bar:
            for batch_idx, (data, label) in enumerate(train_loader):
                data = data.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                output = self(data)
                loss = criterion(output, label)
                loss.backward()
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
                scores = self(data)
                loss = criterion(scores, target)
                # ERROR
                losses.append(loss.cpu()) # loss for each batch
                # save for ploting curve
                class_probs.append(scores.squeeze(-1).cpu())
                class_label.append(target.cpu())
                
        # overall loss
        loss = np.mean(losses)
        return loss, class_probs, class_label