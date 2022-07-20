import os
root_path = os.path.abspath('../..')
train_fpath = root_path + '/data/train_cdhit.txt'
test_fpath = root_path + '/data/test_cdhit.txt'
log_name = 'cnnV1.log'

input_size = 21
in_channels = 1
kernel_size_row=7 
kernel_size_col=21

num_classes = 2
learning_rate = 0.001
batch_size = 10
num_epochs = 1000
step = 1
forward_size = 20

checkpoint_m = 50

model_name='CNNV1'


plot_nEpoch = 10