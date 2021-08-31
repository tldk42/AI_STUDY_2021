import numpy as np

class linear:
    def __init__(self):
        self.W = np.array(np.random.uniform(-10,10)).reshape(1,1)
        self.b = np.array(np.random.uniform(-10,10)).reshape(1,1)

        #   early stop을 위한 변수 step은 over횟수 _loss는 현재 loss
        self._step = 0
        self._loss = float('inf')

    # (참고)  source : https://forensics.tistory.com/29 
    """
    if the network outperforms the previous best model: save a copy of the network at the current epoch
    """
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

    def predict(self, x):
        y = x * self.W.T + self.b
        return y

    def update(self, x, cost, lr):
        delta_w = -(lr*(2/len(cost))*(np.dot(x.T, cost)))
        delta_b = -(lr*(2/len(cost))*np.sum(cost))
        return delta_w, delta_b

    def SGD(self,x, y, iters_num, batch_size, early_stopping = False):   
        
        iter_per_epoch = max(x.shape[0] / batch_size, 1)

        for idx in range(iters_num):
            batch_mask = np.random.choice(x.shape[0], batch_size)
            x = x[batch_mask]
            y = y[batch_mask]

            cost = y - self.predict(x)
            w_delta, b_delta = self.update(x,cost,lr=0.01)
            self.W -= w_delta
            self.b -= b_delta

            if idx % iter_per_epoch == 0:

                if early_stopping is True:
                    #   10 epoch 연속으로 loss 증가시 break
                    if self.early_stop(self.loss(x,y),patience=10):
                        break

                print('[ epoch', int(idx / iter_per_epoch)+1,']')
                print("[ sqe 'cost']")
                print(self.loss(x,y))
                print()

        return self.W, self.b
    
    def loss(self, x, t):
        y = self.predict(x)
        loss = np.sum((y - t)**2)
        return loss / x.shape[0]
