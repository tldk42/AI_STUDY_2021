import numpy as np
from numpy.core.fromnumeric import product
import pickle
def SoftmaxGD(x, y, w, b, learning_rate=0.01, epoch=100000, batch_size=128):
    for epochs in range(epoch):
        batch_mask = np.random.choice(x.shape[0], batch_size)
        x_batch = x[batch_mask]
        y_batch = y[batch_mask]

        z = x_batch.dot(w) + b
        pred = softmax(z)
        dz = (pred - y_batch) / batch_size
        dw = np.dot(x_batch.T, dz)
        db = dz * 1.0
        w -= dw * learning_rate
        b -= (db * learning_rate).mean(0)

        if epochs % (epoch / 10) == 0:
            pred = softmax(x.dot(w) + b)
            print("ACC on epoch %d : " % epochs, (pred.argmax(1) == y.argmax(1)).mean())
            err = cross_entropy_loss(pred, y)
            print("ERR on epoch %d : " % epochs, err)

    return w, b

def softmax(a):
    C = np.max(a)
    exp_a = np.exp(a - C)
    if a.ndim == 1:
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
    else:
        sum_exp_a = np.sum(exp_a, 1)
        sum_exp_a = sum_exp_a.reshape(sum_exp_a.shape[0], 1)
        y = exp_a / sum_exp_a
    return y


def cross_entropy_loss(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y)) / batch_size



# load data
with open('mnist.pkl', 'rb') as f:
    train = pickle.load(f)
x_train = train['train_img']
y_train = train['train_label']
x_train = x_train/ x_train.max()

# y to one-hot
one_hot = np.zeros((y_train.shape[0], y_train.max() + 1))
one_hot[np.arange(y_train.shape[0]), y_train] = 1
y_train = one_hot

# initialize parameters and hyperparameters
num_classes = y_train.shape[1]
w = np.random.uniform(-1, 1, (x_train.shape[1], num_classes))
b = np.zeros(num_classes)

learning_rate = 0.01
epoch = 10000
batch_size = 512

w, b = SoftmaxGD(x_train, y_train, w, b, learning_rate, epoch, batch_size)

pred = x_train.dot(w) + b
pred = softmax(pred)
print("TRAIN ACC : ", (pred.argmax(1) == y_train.argmax(1)).mean())

