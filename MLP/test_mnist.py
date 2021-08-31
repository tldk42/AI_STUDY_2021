from MLP import Model
import pickle
import numpy as np

with open('test_dataset.pkl', 'rb') as f:
    x_test = pickle.load(f)
    y_test = pickle.load(f)

# normalize x(input)
x_test = x_test / x_test.max()

# convert y to one-hot-label
one_hot = np.zeros((y_test.shape[0], y_test.max() + 1))
one_hot[np.arange(y_test.shape[0]), y_test] = 1
y_test = one_hot 

with open('three_layered_params.pkl', 'rb') as f:
    param3 = pickle.load(f)
with open('two_layered_params.pkl', 'rb') as f:
    param2 = pickle.load(f)
with open('one_layered_params.pkl', 'rb') as f:
    param1 = pickle.load(f)


layer3 = [list(param3['w1'][1].shape)[0], list(param3['w2'][1].shape)[0], list(param3['w3'][1].shape)[0]]
layer2 = [list(param3['w1'][1].shape)[0], list(param3['w2'][1].shape)[0]]
layer1 = list(param3['w1'][1].shape)

layer_3 = Model(layer3)
layer_2 = Model(layer2)
layer_1 = Model(layer1)

layer_3.test(x_test, y_test)

