import pickle
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import linear

# squared error
def sqe(y, t):
  loss = np.sum((y - t)**2)
  return loss

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

net = linear()

for idx in range(iters_num):
      # set minibatch
      batch_mask = np.random.choice(train_size, batch_size)
      x_batch = train_x[batch_mask]
      y_batch = train_y[batch_mask]
      
      # SGD algorithum
      grad = net.gradient_descent(x_batch, y_batch)
     
      # result with 1 epoch
      if idx % iter_per_epoch == 0:
        print('[ epoch', int(idx / iter_per_epoch)+1,']')
        print("[ sqe 'W']")
        print(sqe(params['W'], grad[0]))
        print("[ sqe 'b']")
        print(sqe(params['b'], grad[1]))
        print("[ sqe 'cost']")
        print(net.loss(x_batch,y_batch))
        print()

print('[test result]')
print(net.loss(test_x, test_y))

print('[dev result]')
print(net.loss(dev_x, dev_y))


