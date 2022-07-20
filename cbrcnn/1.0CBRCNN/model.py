import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
# processing bar
from tqdm import tqdm
from cbrcnn_hyperparams import *
import numpy as np

from utils import seqSegment

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CBRCNNStage1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropP = 0):
        super(CBRCNNStage1, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                         bidirectional=True)
        
        self.conv11 = nn.Conv2d(in_channels=1, out_channels=1, 
                               kernel_size=(7, hidden_size*2),
                              stride = 1, padding=(3, 0))
        
        self.conv12 = nn.Conv1d(in_channels=1, out_channels=1,
                              kernel_size=1, stride=1, 
                               padding=0)
        
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(p = dropP)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.batch_sizes[0], self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.batch_sizes[0], self.hidden_size).to(device)
        
        packed_out, (hn, cn) = self.lstm1(x,  (h0, c0))
        cx, _ = pad_packed_sequence(packed_out, batch_first=True)
       
        cx = torch.tanh(self.conv11(torch.unsqueeze(cx, 1)))
        
        out = self.conv12(cx.squeeze(3)) 
        out = self.drop(out)
        out = self.sigmoid(out)
        return out
    
class CBRCNNStage2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropP = 0):
        super(CBRCNNStage2, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                         bidirectional=True)
        self.conv11 = nn.Conv2d(in_channels=1, out_channels=1, 
                               kernel_size=(7, hidden_size*2),
                              stride = 1, padding=(3, 0))
        
        self.conv12 = nn.Conv1d(in_channels=1, out_channels=1,
                              kernel_size=1, stride=1, 
                               padding=0)
        
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(p = dropP)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        cx, (hn, cn) = self.lstm1(x,  (h0, c0))
        cx = torch.tanh(self.conv11(torch.unsqueeze(cx, 1)))
        
        out = self.conv12(cx.squeeze(3)) 
        out = self.drop(out)
        out = self.sigmoid(out)
        return out

def train_batch(stage1, stage2, train_loader, optimizerStage1, optimizerStage2, criterion):
    # set training state to model
    stage1.to(device)
    stage2.to(device)
    
    stage1.train()
    stage2.train()

    with tqdm(total=len(train_loader), position=0) as progress_bar:
        for batch_idx, (data, label) in enumerate(train_loader): 
            ###
            # Stage 1 Training
            ###
            data = data.to(device)
            label = label.to(device)
            
            # forward
            optimizerStage1.zero_grad()
            output1 = stage1(data)
            output1 = output1.squeeze(1) # output shape: [batch_size, len_sequence]
            
            label, _ = pad_packed_sequence(label, batch_first=True)
            label = label.squeeze(2) # label shape: [batch_size, len_sequence]
            
            # update weights
            loss1 = criterion(output1, label)
            loss1.backward()
            optimizerStage1.step()
            
            ###
            # Stage 1 Training
            ###
            
            # first, process the input for stage 2
            # Input: segments based on the output of Stage 1
            # Output/Target: stay the same with Stage 1
            
            # Segmentation
            output1 = output1.cpu()
            s2_data = seqSegment(output1).to(device)
            
            # forward
            optimizerStage2.zero_grad()
            output2 = stage2(s2_data)
            output2 = output2.squeeze(1) # output shape: [batch_size, len_sequence]
            
            # update weights
            loss2 = criterion(output2, label)
            loss2.backward()
            optimizerStage2.step()
            
            progress_bar.update(1)

def val_batch(stage1, stage2, val_data, criterion): 
    # set evaluation state to the model
    stage1.to(device)
    stage2.to(device)
    stage1.eval()
    stage2.eval()
    
    s1_losses = []
    s2_losses = []

    s1_class_probs = []
    s2_class_probs = []
    
    class_label = []

    # no gradient needed
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_data):
            data = data.to(device)
            target = target.to(device)
            # forward
            scores1 = stage1(data)
            scores1 = scores1.squeeze(1)

            target, _ = pad_packed_sequence(target, batch_first=True)
            target = target.squeeze(2) # label shape: [batch_size, len_sequence]
            
            loss1 = criterion(scores1, target)
            # ERROR
            s1_losses.append(loss1.cpu()) # loss for each batch
            
            # save for ploting curve
            s1_class_probs.append(scores1.squeeze(-1).cpu())
            class_label.append(target.cpu())
            
            
            # Segmentation
            scores1 = scores1.cpu()
            s2_data = seqSegment(scores1).to(device)
            
            # forward
            scores2 = stage2(s2_data)
            scores2 = scores2.squeeze(1) # output shape: [batch_size, len_sequence]
            
            # update weights
            loss2 = criterion(scores2, target)
            
            # ERROR
            s2_losses.append(loss2.cpu()) # loss for each batch
            
            # save for ploting curve
            s2_class_probs.append(scores2.squeeze(-1).cpu())

    # overall loss
    s1_loss = np.mean(s1_losses)
    s2_loss = np.mean(s2_losses)
    return s1_loss, s2_loss, s1_class_probs, s2_class_probs, class_label