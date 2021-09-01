from MLP import Test
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
with open('four_layered_params.pkl', 'rb') as f:
    param4 = pickle.load(f)
with open('three_layered_params.pkl', 'rb') as f:
    param3 = pickle.load(f)
with open('two_layered_params.pkl', 'rb') as f:
    param2 = pickle.load(f)
with open('one_layered_params.pkl', 'rb') as f:
    param1 = pickle.load(f)
    
layer4 = [list(param3['w1'][1].shape)[0], list(param3['w2'][1].shape)[0], list(param3['w3'][1].shape)[0], list(param3['w4'][1].shape)[0]]
layer3 = [list(param3['w1'][1].shape)[0], list(param3['w2'][1].shape)[0], list(param3['w3'][1].shape)[0]]
layer2 = [list(param3['w1'][1].shape)[0], list(param3['w2'][1].shape)[0]]
layer1 = list(param3['w1'][1].shape)

layer_4 = Test(param4, layer4)
layer_3 = Test(param3, layer3)
layer_2 = Test(param2, layer2)
layer_1 = Test(param1, layer1)

p4 = layer_4.predict(x_test, y_test)
p3 = layer_3.predict(x_test, y_test)
p2 = layer_2.predict(x_test, y_test)
p1 = layer_1.predict(x_test, y_test)

print('Hidden Layer  4')
print('ACC:',p4[0],' Loss' ,p4[1])
print('Hidden Layer  3')
print('ACC:',p3[0],' Loss' ,p3[1])
print('Hidden Layer  2')
print('ACC:',p2[0],' Loss' ,p2[1])
print('Hidden Layer  1')
print('ACC:',p1[0],' Loss' ,p1[1])
