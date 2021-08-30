import pickle
import numpy as np
import Layer

# parameters (w, b)
# loas MNIST dataset
with open('mnist.pkl', 'rb') as f:
    dataset = pickle.load(f)
x_train, y_train = dataset['train_img'], dataset['train_label']
x_test, y_test = dataset['test_img'], dataset['test_label']

x_train = x_train / x_train.max()

# convert y to one-hot-label
one_hot = np.zeros((y_train.shape[0], y_train.max() + 1))
one_hot[np.arange(y_train.shape[0]), y_train] = 1
y_train = one_hot 
one_hot = np.zeros((y_test.shape[0], y_test.max() + 1))
one_hot[np.arange(y_test.shape[0]), y_test] = 1
y_test = one_hot 

# init parameters 
num_classes = y_train.shape[1]
hidden = 50

keys = ['w1', 'b1', 'relu', 'w2', 'b2']
params = {}
params['w1'] = np.random.randn(x_train.shape[1], hidden) / np.sqrt(x_train.shape[1]/2)    # 'he' weight initialize
params['b1'] = np.random.uniform(-1, 1, (hidden))
params['w2'] = np.random.randn(hidden, num_classes) / np.sqrt(hidden/2)    # 'he' weight initialize
params['b2'] = np.random.uniform(-1, 1, (num_classes))

Layers = {}
Layers['w1'] = Layer.MulLayer(params['w1'])
Layers['b1'] = Layer.AddLayer(params['b1'])
Layers['relu'] = Layer.ReluLayer()
Layers['w2'] = Layer.MulLayer(params['w2'])
Layers['b2'] = Layer.AddLayer(params['b2'])
lastlayer = Layer.SoftmaxLayer()

grads = {}

# init hyperparameters
learning_rate = 0.01
epochs = 10000
batch_size = 128

for epoch in range(epochs):
    batch_mask = np.random.choice(x_train.shape[0], batch_size)
    x = x_train[batch_mask]
    y = y_train[batch_mask]

    for key in keys:
        x = Layers[key].forward(x)

    loss = lastlayer.forward(x, y)

    if epoch % (epochs / 10) == 0 :
        pred = Layer.softmax(x)
        print("ACC on epoch %d : " % epoch, (pred.argmax(1) == y.argmax(1)).mean())
        print("LOSS on epoch %d : " % epoch, loss)
    
    dout = lastlayer.backward()

    for key in reversed(keys):
        dout = Layers[key].backward(dout)
        if key != 'relu':
            grads[key] = Layers[key].grad
            params[key] -= learning_rate * grads[key]

