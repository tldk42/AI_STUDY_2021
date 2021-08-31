import pickle
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import linear

#   load datasets
with open('myrandomdataset_ex.pkl','rb') as f:
   params = pickle.load(f)
   dataset_local = pickle.load(f)
   
#  set variable
train_x, train_y = dataset_local['train_x'], dataset_local['train_y']
dev_x, dev_y = dataset_local['dev_x'], dataset_local['dev_y']
test_x, test_y = dataset_local['test_x'], dataset_local['test_y']

#  hyper-parameter
iters_num = 1000
train_size = train_x.shape[0]
batch_size = int(train_size * 0.1)
iter_per_epoch = max(train_size / batch_size, 1)

# train train_dataset
net = linear()
net.SGD(train_x,train_y,iters_num,batch_size,early_stopping=True)

print('[test result]')
print(net.loss(test_x, test_y))

print('[dev result]')
print(net.loss(dev_x, dev_y))


