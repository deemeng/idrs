import os
root_path = os.path.abspath('../..')
train_fpath = root_path + '/data/train_cdhit.txt'
test_fpath = root_path + '/data/test_cdhit.txt'
log_name = 'BiLSTM1.0.log'

input_size = 21
num_layers = 2
hidden_size = 24
num_classes = 2
learning_rate = 0.001
batch_size = 10
num_epochs = 500
step = 1
forward_size = 20


model_name='BiLSTM1.0'

checkpoint_m = 50
plot_nEpoch = 10