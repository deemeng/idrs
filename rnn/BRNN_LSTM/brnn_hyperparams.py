import os
root_path = os.path.abspath('../..')
train_fpath = root_path + '/data/train_cdhit.txt'
test_fpath = root_path + '/data/test_cdhit.txt'

# train_fpath = root_path + '/data/test'
# test_fpath = root_path + '/data/test'

log_name = 'BiLSTM.log'

input_size = 21
num_layers = 4
hidden_size = 24
num_classes = 2
learning_rate = 0.001
batch_size = 5
num_epochs = 200
step = 1
forward_size = 20


model_name='BiLSTM'

checkpoint_m = 10
plot_nEpoch = 1
