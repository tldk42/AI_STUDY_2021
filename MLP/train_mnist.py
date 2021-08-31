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

num_classes = y_train.shape[1]

# init parameters 
"""
        '1 hidden layered newralnet (default)'

    activation_         =  (h) = max(Wx + b, 0) -> RELU
    output_activation_  =  (o) = softmax(Uh + d) 
    loss_               =   J = −yT log(o)
    
"""

Layer_num = [100, 50, 30]       # hidden layer
Layer_len = len(Layer_num)  

model = Model()
# middle(hidden layer)
for idx in range(Layer_len+1):
    #   if hidden layer is with in input layer
    if idx == 0:
        model.addLayer(Layer.MulLayer(), input_size=(x_train.shape[1], Layer_num[0]), name='w1',init='he')
        model.addLayer(Layer.AddLayer(), input_size=Layer_num[0], name='b1')
        model.addLayer(Layer.ReluLayer(), activation=True, name='ReLu1')
    #   if hidden layer is with in ouput layer
    elif idx == Layer_len:
        model.addLayer(Layer.MulLayer(), input_size=(Layer_num[idx-1], y_train.shape[1]), name='w'+str(idx+1),init='he')
        model.addLayer(Layer.AddLayer(), input_size= y_train.shape[1], name='b'+str(idx+1))
        model.addLayer(Layer.SoftmaxLayer(), activation=True, name='softmax')
    #   else   
    else:
        model.addLayer(Layer.MulLayer(), input_size=(Layer_num[idx-1], Layer_num[idx]), name = 'w'+str(idx+1),init='he')
        model.addLayer(Layer.AddLayer(), input_size=Layer_num[idx], name = 'b'+str(idx+1))
        model.addLayer(Layer.ReluLayer(), activation=True, name = 'Relu'+str(idx+1))
# output layer
"""model.addLayer(Layer.MulLayer(), input_size=(x_train.shape[1], Layer_num[0]), name='w1',init='he')
model.addLayer(Layer.AddLayer(), input_size=Layer_num[0], name='b1')
model.addLayer(Layer.ReluLayer(), activation=True, name='ReLu1')
model.addLayer(Layer.MulLayer(), input_size=(Layer_num[0], Layer_num[1]), name = 'w2',init='he')
model.addLayer(Layer.AddLayer(), input_size=Layer_num[1], name = 'b2')
model.addLayer(Layer.ReluLayer(), activation=True, name = 'Relu2')
model.addLayer(Layer.MulLayer(), input_size=(Layer_num[1], y_train.shape[1]), name='w3',init='he')
model.addLayer(Layer.AddLayer(), input_size= y_train.shape[1], name='b3')
model.addLayer(Layer.SoftmaxLayer(), activation=True, name='softmax')
"""
params = model.train(x_train, y_train, 1000, 0.01, int(x_train.shape[0] / 10), early_stopping=True)


#   train 된 w,b pickle로 저장 
"""with open('trained_dataset.pkl','wb') as f:
    pickle.dump(params,f)"""
# 저장된 params 로 test dataset train 결과 