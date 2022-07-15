import os
root_path = os.path.abspath('../..')
train_fpath = root_path + '/data/train_cdhit.txt'
test_fpath = root_path + '/data/test_cdhit.txt'
## For test!!!!!!
# train_fpath = root_path + '/data/train'
# test_fpath = root_path + '/data/test'

model_name='CBRCNN0.0'
log_name = 'CBRCNN0.0.log'
input_size = 21
dropP = 0
num_classes = 2
learning_rate = 0.001
num_epochs = 200
checkpoint_m = 100
plot_nEpoch = 10
batch_size = 16
# rnn pramas
num_layers = 2
hidden_size = 24

# cnn pramas
# not using
in_channels = 1
kernel_size_row=7 
kernel_size_col=21
