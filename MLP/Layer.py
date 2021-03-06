import numpy as np

"""
    functions and class (layer) for MLP model
    
    cross entropy loss
    softmax
    * layer
    + layer
    softmax layer 
    relu layer
"""

def cross_entropy_loss(y, t):
        c = 1e-7
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        batch_size = y.shape[0]
        return -np.sum(t * np.log(y + c)) / batch_size

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
    
class MulLayer:
    def __init__(self, param=None):
        self.x = None
        self.param = param
        self.grad = None

    def forward(self, x):
        self.x = x
        return x.dot(self.param)

    def backward(self, dout):
        self.grad = np.dot(self.x.T, dout)
        return np.dot(dout, self.param.T)

class AddLayer:
    def __init__(self, param=None):
        self.x = None
        self.param = param
        self.grad = None

    def forward(self, x):
        self.x = x
        return x + self.param

    def backward(self, dout):
        self.grad = dout.mean()
        return dout

class SoftmaxLayer:
    def __init__(self):
        self.loss = None
        self.y = None
        self.y = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_loss(self.y, self.t)
        return self.loss

    def backward(self):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / (batch_size)
        return dx

class ReluLayer:
    def __init__(self):
        self.out = None
    
    def forward(self, z):
        self.out = z
        self.out[self.out <= 0] = 0
        return self.out
    
    def backward(self, dout):
        self.out[self.out > 0] = 1
        return self.out * dout 

