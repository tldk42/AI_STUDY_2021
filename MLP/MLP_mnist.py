from os import name
import numpy as np
from MLP import Model
import pickle
import Layer

# load (train)dataset   ,,  지금 작업하고 있는 장치에서 scikit의 mnist 데이터셋을 불러오는데에 문제가 생김 --> 직접 mnist.pkl 다운로드
with open('train_dataset.pkl', 'rb') as f:
    x_train = pickle.load(f)
    y_train = pickle.load(f)

# normalize x(input)
x_train = x_train / x_train.max()

# convert y to one-hot-label
one_hot = np.zeros((y_train.shape[0], y_train.max() + 1))
one_hot[np.arange(y_train.shape[0]), y_train] = 1
y_train = one_hot 

num_classes = y_train.shape[1]

# init parameters 
"""
        'hidden layered newralnet (default=1, it can be 2 or more)'

    activation_         =  (h) = max(Wx + b, 0) -> RELU
    output_activation_  =  (o) = softmax(Uh + d) 
    loss_               =   J = −yT log(o)
    
"""

Layer_num = [100, 50]       # hidden layer
Layer_len = len(Layer_num)  

model = Model()
# middle(hidden layer)
for idx in range(Layer_len+1):

    #   if hidden layer is with in input layer  (relu)
    if idx == 0:
        model.addLayer(Layer.MulLayer(), input_size=(x_train.shape[1], Layer_num[0]), name='w1',init='he')
        model.addLayer(Layer.AddLayer(), input_size=Layer_num[0], name='b1')
        model.addLayer(Layer.ReluLayer(), activation=True, name='ReLu1')

    #   if hidden layer is with in ouput layer  (softmax)
    elif idx == Layer_len:
        model.addLayer(Layer.MulLayer(), input_size=(Layer_num[idx-1], y_train.shape[1]), name='w'+str(idx+1),init='he')
        model.addLayer(Layer.AddLayer(), input_size= y_train.shape[1], name='b'+str(idx+1))
        model.addLayer(Layer.SoftmaxLayer(), activation=True, name='softmax')

    #   else
    else:
        model.addLayer(Layer.MulLayer(), input_size=(Layer_num[idx-1], Layer_num[idx]), name = 'w'+str(idx+1),init='he')
        model.addLayer(Layer.AddLayer(), input_size=Layer_num[idx], name = 'b'+str(idx+1))
        model.addLayer(Layer.ReluLayer(), activation=True, name = 'Relu'+str(idx+1))


#   start train
params = model.train(x_train, y_train, epoch=100, learning_rate=0.01,batch_size=500, early_stopping=True, patience=5)

#   train 된 w,b pickle로 저장 
if Layer_len == 1:
    with open('one_layered_params.pkl','wb') as f:
        pickle.dump(params,f)
elif Layer_len ==2:
    with open('two_layered_params.pkl','wb') as f:
        pickle.dump(params,f)
elif Layer_len ==3:
    with open('three_layered_params.pkl','wb') as f:
        pickle.dump(params,f)
else:
    with open('four_layered_params.pkl','wb') as f:
        pickle.dump(params,f)

# 저장된 params 로 test dataset train 결과 