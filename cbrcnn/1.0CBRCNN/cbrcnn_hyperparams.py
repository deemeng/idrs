import os
root_path = os.path.abspath('../..')
train_fpath = root_path + '/data/train_cdhit.txt'
test_fpath = root_path + '/data/test_cdhit.txt'
## For test!!!!!!
# train_fpath = root_path + '/data/train'
# test_fpath = root_path + '/data/test'

model_name='CBRCNN1.0'
log_name = 'CBRCNN1.0.log'

input_size = 21
input_size_stage2 = 15

dropP = 0
num_classes = 2
learning_rate = 0.001
num_epochs = 500
checkpoint_m = 100
plot_nEpoch = 10
batch_size = 1
# rnn pramas
num_layers = 1

hidden_size = 40
hidden_size_stage2 = 40

# cnn pramas
# not using yet
in_channels = 1
kernel_size_conv1=7
kernel_size_conv2=1
