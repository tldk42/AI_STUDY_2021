from logistic_regression import Logistic
import numpy as np
import pickle

# load mnist dataset (sklearn dataset에서 데이터를 가져올 때 계속 오류가 생겨 직접 다운로드 받았습니다.)
with open("mnist.pkl", 'rb') as f:
    dataset = pickle.load(f)

x_train = dataset['train_img']
y_train = dataset['train_label']
x_test = dataset['test_img']
y_test = dataset['test_label']

# normalize dataset
x_train = x_train / x_train.max()
x_test = x_test / x_test.max()

# y_train to one-hot
one_hot = np.zeros((y_train.shape[0], y_train.max() + 1))
one_hot[np.arange(y_train.shape[0]), y_train] = 1
y_train = one_hot
one_hot = np.zeros((y_test.shape[0], y_test.max() + 1))
one_hot[np.arange(y_test.shape[0]), y_test] = 1
y_test = one_hot

net  = Logistic()
w, b = net.SGD(x_train,y_train, early_stopping=True)

pred = x_test.dot(w) + b
pred = net.softmax(pred)
print('------------------------')
print("TEST ACC : ", (pred.argmax(1) == y_test.argmax(1)).mean())
