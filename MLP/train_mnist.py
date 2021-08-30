from os import name
import numpy as np
from Model import Model
import pickle
import Layer

# load data
with open('train_dataset.pkl', 'rb') as f:
    x_train = pickle.load(f)
    y_train = pickle.load(f)

x_train = x_train / x_train.max()

# convert y to one-hot-label
one_hot = np.zeros((y_train.shape[0], y_train.max() + 1))
one_hot[np.arange(y_train.shape[0]), y_train] = 1
y_train = one_hot 

# init parameters 
num_classes = y_train.shape[1]
Layer_num = [100, 50]
model = Model()
model.addLayer(Layer.MulLayer(), input_size=(x_train.shape[1], Layer_num[0]), name='w1',init='he')
model.addLayer(Layer.AddLayer(), input_size=Layer_num[0], name='b1',init='he')
model.addLayer(Layer.ReluLayer(), activation=True, name='ReLu1')
model.addLayer(Layer.MulLayer(), input_size=(Layer_num[0], Layer_num[1]), name = 'w2',init='he')
model.addLayer(Layer.AddLayer(), input_size=Layer_num[1], name = 'b2',init='he')
model.addLayer(Layer.ReluLayer(), activation=True, name = 'Relu2')
model.addLayer(Layer.MulLayer(), input_size=(Layer_num[1], y_train.shape[1]), name='w3',init='he')
model.addLayer(Layer.AddLayer(), input_size= y_train.shape[1], name='b3',init='he')
model.addLayer(Layer.SoftmaxLayer(), activation=True, name='softmax')

params = model.train(x_train, y_train, 10000, 0.01, 128)


#   train 된 w,b pickle로 저장 
with open('trained_dataset.pkl','wb') as f:
    pickle.dump(params,f)
# 저장된 params 로 test dataset train 결과 