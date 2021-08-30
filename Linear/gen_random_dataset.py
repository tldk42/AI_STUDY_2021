import numpy as np
import pickle

#   init true_paremeters
params = {}
params['W'] = np.array(np.random.uniform(-10,10)).reshape(1,1)
params['b'] = np.array(np.random.uniform(-10,10)).reshape(1,1)

#   init dataset
N = 10000
x = np.array(np.random.uniform(-10,10, N)).reshape(N,1)
y = np.array(np.random.normal((x * params['W'] + params['b']), 1)).reshape(N,1)

#   divide dataset
train_size = int(N * 0.85)
dev_size = int(N * 0.05)
test_size = int(N * 0.1)

dataset = {}
dataset['train_x'] = x[:train_size]
dataset['train_y'] = y[:train_size]
dataset['dev_x'] = x[-(dev_size + test_size):-test_size]
dataset['dev_y'] = y[-(dev_size + test_size):-test_size]
dataset['test_x'] = x[-test_size:]
dataset['test_y'] = y[-test_size:]

#   save dataset
with open('myrandomdataset_ex.pkl', 'wb') as f:
    pickle.dump(params,f)
    pickle.dump(dataset,f)

