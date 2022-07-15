import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
# processing bar
from tqdm import tqdm
from cbrcnn_hyperparams import *
import numpy as np

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CBRCNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropP = 0):
        super(CBRCNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                         bidirectional=True)
        # self.fc = nn.Linear(hidden_size*2, 1)
        
        self.conv11 = nn.Conv2d(in_channels=1, out_channels=3, 
                               kernel_size=(7, hidden_size*2),
                              stride = 1, padding=(3, 0))
        
#         self.conv12 = nn.Conv2d(in_channels=7, out_channels=1, 
#                                kernel_size=(3, 3),
#                               stride = 1, padding=(1, 1))
        
        self.conv12 = nn.Conv1d(in_channels=3, out_channels=1,
                              kernel_size=1, stride=1, 
                               padding=0)
        
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(p = dropP)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.batch_sizes[0], self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.batch_sizes[0], self.hidden_size).to(device)
        # Forward Prop
        # out, (hn, cn) = self.lstm(x,  (h0, c0))
        # out, _ = self.lstm(x,  (h0, c0))
        packed_out, (hn, cn) = self.lstm1(x,  (h0, c0))
        cx, _ = pad_packed_sequence(packed_out, batch_first=True)
        # all training example, last hidden state, all 
        # it is not last hidden state, it is the last batch
        # print('out.squeeze() ', out.squeeze().size())
        # out = self.fc(out)
        
        cx = torch.tanh(self.conv11(torch.unsqueeze(cx, 1)))
        
        out = self.conv12(cx.squeeze(3)) 
        out = self.drop(out)
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
                output = output.squeeze().unsqueeze(2)
                
                label, _ = pad_packed_sequence(label, batch_first=True)
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
                scores = scores.squeeze().unsqueeze(2)
                
                target, _ = pad_packed_sequence(target, batch_first=True)
                
                loss = criterion(scores, target)
                # ERROR
                losses.append(loss.cpu()) # loss for each batch
                # save for ploting curve
                class_probs.append(scores.squeeze(-1).cpu())
                class_label.append(target.cpu())
                
        # overall loss
        loss = np.mean(losses)
        return loss, class_probs, class_label