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

num_classes = y_test.shape[1]


with open('three_layered_params.pkl', 'rb') as f:
    param3 = pickle.load(f)


