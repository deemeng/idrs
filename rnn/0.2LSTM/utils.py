'''
This .py files contain all util functions:

1. load_dataset
2. get_num_class

'''
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import numpy as np

# import hyperparameters
from brnn_hyperparams import *

# plots
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score

def load_dataset(fpath):
    '''
    params:
        fpath - path to fasta file, training or test
                ***The format of the sets:

                The first two lines are a global header:
                total_number_of_records
                input_size number_of_classes

                total_number_of_records is simply the number of proteins in the set,
                input_size is how many numbers are used to represent one amino acid (21
                or 22, probably, in your case) and number_of_classes is the number of
                classes...

                After that you have the proteins, 5 lines each:

                line 1: name of the protein
                line 2: number of amino acids in the protein
                line 3: input
                line 4: targets
                line 5: empty

                You can use any name you want so long as it's unique, a single word, and
                not outrageously long.

                The input should be a single long list of the numbers representing the
                amino acids in the protein. If you use, say, 21 numbers per amino acid,
                and the protein is 100 amino acids long, the input line will contain
                2100 numbers, with the first 21 being the representation of the first
                amino acid in the protein, the following 21 the representation of the
                second, etc.
                For the moment (before we use alignments) the representation of an amino
                acid will be a one-hot encoding, e.g.:

                A     -> 1 0 0 0 ..... 0 0
                C     -> 0 1 0 0 ..... 0 0
                ...
                Y     -> 0 0 0 0 ..... 1 0
                other -> 0 0 0 0 ..... 0 1

                where "other" is unknown or weird amino acid (X, B, J, O, U, Z)

                The line containing the targets is a list of integers representing the
                classes of the amino acids. There are as many integers as there are
                amino acids in the protein. You can choose whatever integers you want,
                but it's probably simplest to have something like class1=0, class2=1,
                class3=2, etc..

                (notice that in the sample sets in the directory you have a more
                complicated representation of the inputs, where there are a lot of
                floating point numbers rather than just 0 and 1, and that's because
                those inputs are frequency profiles from MSA - so you can see how the
                code works for both kinds of inputs)
                
    returns:
        p_data - list[data_tensor, target_tensor]
        p_lens - list, protein length
    
    Note: 
        - the reason not using tensor to save protein Sequences and Targets is we have varying length sequences! 
        - solve this problem we could consider pading. But our dataset lens range from about 20 to 10,000. 
            Thus, padding maybe not a good idea here.
    '''
    num_protein = 0
    num_i = 0
    num_o = 0

    # p_names = []
    p_lens = []
    # p_seqs = []
    # p_anns = []
    p_data = []
    with open(fpath) as fp:
        num_protein = int(fp.readline())
        num_io = fp.readline().split(' ')
        num_i = int(num_io[0])
        num_o = int(num_io[0])

        line = fp.readline()
        while line:
            # p_name = line[:-1]
            p_len = int(fp.readline())
            p_sequence = torch.tensor([int(x) for x in fp.readline().split(' ')], 
                                      dtype=torch.float32).reshape(-1, 21)
            p_annotation = torch.tensor([[int(x)] for x in fp.readline().split(' ')], dtype=torch.float32)
            # skip empty
            next(fp)
            # p_names.append(p_name)
            p_lens.append(p_len)
            # p_seqs.append(p_sequence)
            # p_anns.append(p_annotation)
            p_data.append([p_sequence, p_annotation])
            line = fp.readline()   
    return p_data, p_lens
"""
Custumerized Dataset
"""
class ProteinDataset(Dataset):
    
    def __init__(self, fpath, transform=None):
        self.p_data, self.p_lens = load_dataset(fpath)
    
    def __len__(self) -> int:
        return len(self.p_lens)
    
    def __getitem__(self, i) -> torch.Tensor:
        # self.p_data[i][0], sequence;  self.p_data[i][1], label
        return self.p_data[i][0], self.p_data[i][1]
    def numAAs(self) -> int:
        return sum(self.p_lens)
"""
Variable-length sequences padding method
pad_sequence & pack_padded_sequence & pad_packed_sequence
"""
def pad_packed_collate(batch):
    """Puts data, and lengths into a packed_padded_sequence then returns
       the packed_padded_sequence and the labels. Set use_lengths to True
       to use this collate function.
       Args:
         batch: (list of tuples) [(sequence, target)].
             sequence is a FloatTensor
             target has the same variable length with sequence
       Output:
         packed_batch: (PackedSequence), see torch.nn.utils.rnn.pack_padded_sequence
         labels: (Tensor), labels from the file names of the wav.
    """

    if len(batch) == 1:
        seqs, labels = [batch[0][0]], [batch[0][1]]
        lengths = [seqs[0].size(0)]
        
    if len(batch) > 1:
        # get data and sorted by the length of sequence
        seqs, labels, lengths = zip(*[(a, b, a.size(0)) for (a,b) in sorted(batch, key=lambda x: x[0].size(0), reverse=True)])
    seqs = pad_sequence(seqs, batch_first=True)
    labels = pad_sequence(labels, batch_first=True)
    packed_seqs = pack_padded_sequence(seqs, lengths, batch_first=True)
    packed_labels = pack_padded_sequence(labels, lengths, batch_first=True)
    
    return packed_seqs, packed_labels

def get_num_class(input_data):
    '''
    params:
        input_data - p_data format return from load_dataset function.
        
    returns:
        c0 - the number of amino acids of class 0, which means ordered.
        c1 - the number of amino acids of class 1, which means disordered.
    '''
    c0 = 0
    c1 = 0
    for batch_idx, (data, target) in enumerate(input_data):
        c0 = c0 + len(target) - target.sum()
        c1 = c1 + target.sum()
    return c0, c1

# Plot roc curve
def rocPlot(train_label, train_probs, val_label, val_probs):
    fig = plt.figure(figsize=(8, 8))
    
    t_fpr, t_tpr, t_thresholds = roc_curve(train_label, train_probs)
    t_roc_auc = auc(t_fpr, t_tpr)
    v_fpr, v_tpr, v_thresholds = roc_curve(val_label, val_probs)
    v_roc_auc = auc(v_fpr, v_tpr)
    
    lw = 2
    plt.plot(
        t_fpr,
        t_tpr,
        color="darkorange",
        lw=lw,
        label="Training (area = %0.2f)" % t_roc_auc,
    )
    
    plt.plot(
        v_fpr,
        v_tpr,
        color="darkgreen",
        lw=lw,
        label="Val (area = %0.2f)" % v_roc_auc,
    )
    
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(model_name+": Train & Val ROC curve")
    plt.legend(loc="lower right")
    
    plt.show()
    return fig