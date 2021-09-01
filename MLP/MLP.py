from Layer import *
import numpy as np

# for train model 
class Model:
    def __init__(self, layer):
        self.layer = layer

        self.params = {}
        self.grads = {}
        self.keys = []
        self.layers = {}
        self.num = 0
        self.loss = None
        self.pred = None
        self._step = 0
        self._loss = float('inf')


        for idx in range(len(layer)+1):
            
            #   if hidden layer is with in input layer  (relu)
            if idx == 0:
                self.addLayer(MulLayer(), input_size=(784, layer[0]), name='w1',init='he')
                self.addLayer(AddLayer(), input_size=layer[0], name='b1')
                self.addLayer(ReluLayer(), activation=True, name='ReLu1')

            #   if hidden layer is with in ouput layer  (softmax)
            elif idx == len(layer):
                self.addLayer(MulLayer(), input_size=(layer[idx-1], 10), name='w'+str(idx+1),init='he')
                self.addLayer(AddLayer(), input_size= 10, name='b'+str(idx+1))
                self.addLayer(SoftmaxLayer(), activation=True, name='softmax')

            #   else
            else:
                self.addLayer(MulLayer(), input_size=(layer[idx-1], layer[idx]), name = 'w'+str(idx+1),init='he')
                self.addLayer(AddLayer(), input_size=layer[idx], name = 'b'+str(idx+1))
                self.addLayer(ReluLayer(), activation=True, name = 'Relu'+str(idx+1))
    
    def addLayer(self, layer, activation = False, input_size=None, name=None, init=None):
        if name is None:
            name = str(self.num)
        
        self.keys.append(name)
        self.num += 1
        self.layers[name] = layer

        if not activation:
            if isinstance(layer, AddLayer):
                self.params[name] = np.zeros(input_size)
            elif init == 'he':
                n = np.sqrt(6 / input_size[0])
                #self.params[name] = np.random.uniform(-n, n, input_size)
                self.params[name] = np.random.randn(input_size[0], input_size[1]) * np.sqrt(2/input_size[0])
            else:
                self.params[name] = np.random.uniform(-1, 1, input_size)
            self.layers[name].param = self.params[name]
        
    def predict(self ,x ,y):
        for i in range(len(self.keys) - 1):
            key = self.keys[i]
            x = self.layers[key].forward(x)
        self.loss = self.layers[self.keys[-1]].forward(x, y)
        self.pred = softmax(x)

    def early_stop(self, loss ,patience=10, verbose=1):
        if self._loss < loss:
            self._step += 1
            if self._step > patience:
                if verbose:
                    print('Training process early stopped..!')
                return True
        else:
            self._step = 0
            self._loss = loss

        return False

    def train(self, x_train, y_train, epoch, learning_rate, batch_size, early_stopping=False, patience=10):

        iters_per_epochs = max(x_train.shape[0] / batch_size , 1)   # whole dataset / batch dataset  .. ex) 60000/100 -> 600

        for epochs in range(int(epoch * iters_per_epochs)):
            batch_mask = np.random.choice(x_train.shape[0], batch_size)
            x = x_train[batch_mask]
            y = y_train[batch_mask]

            self.predict(x, y)
            dout = self.layers[self.keys[-1]].backward()
            for i in reversed(range(len(self.keys) - 1)):
                key = self.keys[i]
                dout = self.layers[key].backward(dout)
                if key in self.params:
                    self.grads[key] = self.layers[key].grad
                    self.params[key] -= learning_rate * self.grads[key]
                
            if epochs % iters_per_epochs == 0:
                if early_stopping :
                    if self.early_stop(self.loss, patience):
                        break
                print("ACC on epoch", int(epochs / iters_per_epochs)+1, ": ", (self.pred.argmax(1) == y.argmax(1)).mean())
                print("LOSS on epoch", int(epochs / iters_per_epochs)+1, ": ", self.loss)

        return self.params
    def test(self, x, y):
            
        for idx in range(x.shape[0]):
            
            self.predict(x,y)

 
                
            print(self.loss)

# for test
class Test:
    def __init__(self, params, layer):
        self.params = params
        self.grads = {}
        self.keys = []
        self.layers = {}
        self.num = 0
        self.loss = None
        self.pred = None
        self.layer = layer

        for idx in range(len(layer)+1):
            
            #   if hidden layer is with in input layer  (relu)
            if idx == 0:
                self.addLayer(MulLayer(), input_size=(784, layer[0]), name='w1',init='he')
                self.addLayer(AddLayer(), input_size=layer[0], name='b1')
                self.addLayer(ReluLayer(), activation=True, name='ReLu1')

            #   if hidden layer is with in ouput layer  (softmax)
            elif idx == len(layer):
                self.addLayer(MulLayer(), input_size=(layer[idx-1], 10), name='w'+str(idx+1),init='he')
                self.addLayer(AddLayer(), input_size= 10, name='b'+str(idx+1))
                self.addLayer(SoftmaxLayer(), activation=True, name='softmax')

            #   else
            else:
                self.addLayer(MulLayer(), input_size=(layer[idx-1], layer[idx]), name = 'w'+str(idx+1),init='he')
                self.addLayer(AddLayer(), input_size=layer[idx], name = 'b'+str(idx+1))
                self.addLayer(ReluLayer(), activation=True, name = 'Relu'+str(idx+1))

    def addLayer(self, layer, activation = False, input_size=None, name=None, init=None):
        if name is None:
            name = str(self.num)
        
        self.keys.append(name)
        self.num += 1
        self.layers[name] = layer

        if not activation:
            self.layers[name].param = self.params[name]

    def predict(self ,x ,y):
        for i in range(len(self.keys) - 1):
            key = self.keys[i]
            x = self.layers[key].forward(x)
        self.loss = self.layers[self.keys[-1]].forward(x, y)
        self.pred = softmax(x)
        acc = (self.pred.argmax(1) == y.argmax(1)).mean()
        return acc, self.loss