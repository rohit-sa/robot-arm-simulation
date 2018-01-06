# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 00:03:24 2017

@author: Rohit
"""

    
from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Input,Dense
from keras import optimizers

from pathlib import Path
import matplotlib.pyplot as plt
import keras.backend as K
import numpy as np
import pandas as pd

"""
function " neural_net "
Notes:creates Artificial neural network to fit to X
returns the trained model. May re-train previous model
"""
def neural_net(X,y,y_class,prev_model=False):
    """
    PARAMETER SETTINGS
    slight preprocessing
    """
    y_labels = [y]
    layer_param = dict(kernel_initializer='truncated_normal',activation='tanh',
                       bias_initializer='ones')
    
    for i in range(y_class.shape[1]):
        y_labels.append(y_class[:,i,:])
    nn_name = Path('model.hd5')
    
    
    if prev_model & nn_name.exists():
        print('Using prev model')
        nn_architecture = load_model('model.hd5')
    else:
       
        inputs = Input(shape=(X.shape[1],))
        x = Dense(100,**layer_param)(inputs)
#        x = Dense(20,**layer_param)(x)
        output_0 = Dense(y_class.shape[1],activation='linear',name='output_0')(x)
        output_1 = Dense(y_class.shape[2],activation='softmax',name='output_1')(x)
        output_2 = Dense(y_class.shape[2],activation='softmax',name='output_2')(x)
        output_3 = Dense(y_class.shape[2],activation='softmax',name='output_3')(x)
        output_4 = Dense(y_class.shape[2],activation='softmax',name='output_4')(x)
        nn_architecture = Model(inputs=inputs,outputs=[output_0,output_1,output_2,
                                                       output_3,output_4])
#        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#        nn_architecture.compile(loss='mean_squared_error', optimizer=sgd)
        nn_architecture.compile(optimizer='adam',
                                metrics={'output_0':'mean_squared_error','output_1':'accuracy',
                                      'output_2':'accuracy','output_3':'accuracy',
                                      'output_4':'accuracy'},
                                loss={'output_0':'mean_squared_error','output_1':'categorical_crossentropy',
                                      'output_2':'categorical_crossentropy','output_3':'categorical_crossentropy',
                                      'output_4':'categorical_crossentropy'},
                                loss_weights={'output_0':1.0,'output_1': 0.5, 'output_2': 0.2,
                                              'output_3': 0.5, 'output_4': 0.3})
    
    history = nn_architecture.fit(X,y_labels,verbose=2,epochs=200,shuffle=True,
                                  batch_size=64,validation_split=0.1)

    loss = history.history['loss']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(len(loss)),loss)
    plt.show()
    return nn_architecture
"""
function " transform "
Notes: converts x,y,z positions to r, theta and phi
coordiante system
"""
def transform(X):
    r = np.linalg.norm(X,axis=1)
    theta = (np.pi/2) - np.arccos(X[:,1]/r)
    phi = np.arctan2(X[:,2],X[:,0])
    return np.vstack((r,theta,phi)).T 
"""
function " proj_plane "
Notes: defines xz as a single plane reducing dimensionality to 2
removing one output angle
"""
def proj_plane(X):
    x,y,z = X[:,0], X[:,1], X[:,2]
    xz = np.sqrt(x*x + z*z)
    return np.vstack((xz,y)).T
"""
function " proj_plane "
Notes: defines xz as a single plane reducing dimensionality to 2
removing one output angle
"""
def one_hot_encoder(X,N=10):
    one_hot_X = np.zeros((X.shape[0],N))
    for i in range(X.shape[0]):
        index = int(X[i])
        one_hot_X[i,index] = 1
    return one_hot_X
"""
function " proj_plane "
Notes: defines xz as a single plane reducing dimensionality to 2
removing one output angle
"""
def create_drawers(y,N=10):
    y_class = []
    bin_params = []
    for i in range(y.shape[1]):
        min_val = min(y[:,i])
        max_val = max(y[:,i])
        drawers = ((N-1)*(y[:,i]-min_val))//(max_val-min_val)
        bin_params.append({'min_val':min_val,'max_val':max_val})
        y_class.append(one_hot_encoder(drawers,N))
    y_class = np.array(y_class).reshape((y.shape[0],y.shape[1],N))
    bin_params.append({'total_drawers':N})
    return y_class, bin_params
"""
function " proj_plane "
Notes: defines xz as a single plane reducing dimensionality to 2
removing one output angle
"""
def normalize_lengths(X):
    arm_param = {'x_span':(3,15),'y_span':(-6,18),'z_span':(-15,15)}
    X[:,0] = (X[:,0]-arm_param['x_span'][0])/(arm_param['x_span'][1]-arm_param['x_span'][0])
    X[:,1] = (X[:,1]-arm_param['x_span'][0])/(arm_param['x_span'][1]-arm_param['x_span'][0])
    X[:,2] = (X[:,2]-arm_param['x_span'][0])/(arm_param['x_span'][1]-arm_param['x_span'][0])
    return X

"""
<<<<<<< Updated upstream
function: load_data
input parameters: None
Notes: 
=======
function " proj_plane "
Notes: defines xz as a single plane reducing dimensionality to 2
removing one output angle
>>>>>>> Stashed changes
"""
def load_data():
    data = pd.read_csv('dataset.csv')
    out_cols = [col for col in data.columns if 'angle' in col]
#    print(data.describe())
    
    y = data[out_cols].values
    y_class, bin_params = create_drawers(y,20)
    y = np.radians(y)
    data.drop(out_cols,axis=1,inplace=True)
    
    X = data.values
    row,col = X.shape
    X_norm = X
    X_norm = normalize_lengths(X)
#    X_norm = transform(X)
#    o = np.sum(y,axis=1)
#    X_norm = np.column_stack((X_norm,o))
    
    train_index = np.random.choice(row,int(0.85*row),replace=False)
    test_index = np.setdiff1d(range(row),train_index,assume_unique=True)
    
    X_train = X_norm[train_index]
    y_train = y[train_index]
    y_class_train = y_class[train_index]

    X_test = X_norm[test_index]
    y_test = y[test_index]
    y_class_test = y_class[test_index]
    
    return X_train, y_train, y_class_train, X_test, y_test, y_class_test, bin_params

#def custom_loss(y_true,y_pred):
    

"""
function " main "
Notes: load dataset, transform values
train model and save
"""   
def main():
    X_train, y_train, y_class_train, X_test, y_test, y_class_test, bin_params = load_data()
    nn = neural_net(X_train,y_train,y_class_train,False)
    nn.save('model.hd5')
    y_pred = nn.predict(X_test)
    print(y_pred[0])
    print(np.degrees(y_pred[0]))
    print()
    print(np.degrees(y_test))
    print()
    res = pd.DataFrame(np.degrees(y_pred[0] - y_test))
    print(res.describe())
    print(np.degrees(np.mean(abs(y_pred[0]-y_test),axis=0)))

def convert_to_label(y_pred):
    y = []
    for i in range(4):
        value = np.argmax(y_pred[i])
        y.append(value)
    return y

<<<<<<< Updated upstream
"""
function: upack_drawers
input parameters: y_class
Notes: 
"""
def upack_drawers(y_class):
=======
def upack_drawers(y_pred,bin_params):
>>>>>>> Stashed changes
    y = []
    N = bin_params[-1]['total_drawers']-1
    for i in range(4):
        value = np.argmax(y_pred[i])
        min_value = bin_params[i]['min_val']
        max_value = bin_params[i]['max_val']
        y.append((value*(max_value-min_value)/N) + min_value)
    return y

"""
function: testing
input parameters: None
Notes: 
"""
def testing():
    X_train, y_train, y_class_train, X_test, y_test, y_class_test, bin_params = load_data()
    nn = load_model('model.hd5')
    y_pred = nn.predict(X_test[:1,:])
    y_pred_reg = y_pred[0]
    y_pred_class = y_pred[1:]
    print(np.degrees(y_pred_reg))
    print(np.degrees(y_test[0])) 
    print(convert_to_label(y_pred_class))
    print(convert_to_label(y_class_test[0]))
    print(upack_drawers(y_pred_class,bin_params))
    print(upack_drawers(y_class_test[0],bin_params))
    y_pred = nn.predict(X_test)
    print(y_pred)
    res = pd.DataFrame(np.degrees(abs(y_pred[0] - y_test)))
    print(res.describe())
    print(np.degrees(np.mean(abs(y_pred[0]-y_test),axis=0)))
    
if __name__ == '__main__':
    main()
#    testing()