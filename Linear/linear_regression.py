import numpy as np

class linear:
    def __init__(self):
        self.W = np.array(np.random.uniform(-10,10)).reshape(1,1)
        self.b = np.array(np.random.uniform(-10,10)).reshape(1,1)

    def predict(self, x):
        y = x * self.W.T + self.b
        return y

    def update(self, x, cost, lr):
        delta_w = -(lr*(2/len(cost))*(np.dot(x.T, cost)))
        delta_b = -(lr*(2/len(cost))*np.sum(cost))
        return delta_w, delta_b

    def gradient_descent(self,x, y, early_stopping = False):   
        cost = y - self.predict(x)
        w_delta, b_delta = self.update(x,cost,lr=0.01)
        self.W -= w_delta
        self.b -= b_delta

        if early_stopping is True:
            """
            if the network outperforms the previous best model: save a copy of the network at the current epoch
            """

        return self.W, self.b
    
    def loss(self, x, t):
        y = self.predict(x)
        loss = np.sum((y - t)**2)
        return loss / x.shape[0]
