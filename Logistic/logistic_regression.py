import numpy as np

class Logistic:
    def __init__(self):
        self.W = np.random.randn(784, 10) / np.sqrt(784/2)    # 'he' weight initialize
        self.b = 0

    def softmax(self,a):
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

    def predict(self, x):
        a = np.dot(x, self.W) + self.b
        z = self.softmax(a)

        return z

    def loss(self, x, t):
        y = self.predict(x)

        return self.cross_entropy_loss(y,t)

    def cross_entropy_loss(self,y, t):
        c = 1e-7
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        batch_size = y.shape[0]
        return -np.sum(t * np.log(y + c)) / batch_size

    def SoftmaxGD(self, x, y, learning_rate=0.01, epoch=100, batch_size=100):
        
        w, b = self.W, self.b
        iters_per_epochs = max(x.shape[0] / batch_size , 1)

        for epochs in range(int(epoch * iters_per_epochs)):
            batch_mask = np.random.choice(x.shape[0], batch_size)
            x_batch = x[batch_mask]
            y_batch = y[batch_mask]

            z = x_batch.dot(w) + b
            pred = self.softmax(z)
            dz = (pred - y_batch) / batch_size
            dw = np.dot(x_batch.T, dz)
            db = dz * 1.0
            w -= dw * learning_rate
            b -= (db * learning_rate).mean(0)

            if epochs % iters_per_epochs == 0:
                print('[epoch', int(epochs / iters_per_epochs)+1,']')
                pred = self.softmax(x.dot(w) + b)
                acc = (pred.argmax(1) == y.argmax(1)).mean()
                print("ACC : ", acc)
                err = self.cross_entropy_loss(pred, y)
                print("ERR : ", err)

        return w, b
