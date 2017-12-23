# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 00:03:24 2017

@author: Rohit
"""

    
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
function " neural_net "
Notes:creates Artificial neural network to fit to X
returns the trained model. May re-train previous model
"""
def neural_net(X,y,prev_model=False):
    nn_name = Path('model.hd5')
    if prev_model & nn_name.exists():
        print('Using prev model')
        nn_architecture = load_model('model.hd5')
    else:
        nn_architecture = Sequential()
        layer_param = dict(kernel_initializer='truncated_normal',
                           activation='relu',
                           bias_initializer='ones')
        nn_architecture.add(Dense(100, input_shape=(3,), **layer_param))
#        nn_architecture.add(Dense(15, **layer_param))
        nn_architecture.add(Dense(y.shape[1],activation='sigmoid'))
        
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#        nn_architecture.compile(loss='mean_squared_error', optimizer=sgd)
        nn_architecture.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = nn_architecture.fit(X,y,verbose=2,epochs=250,shuffle=True,batch_size=100,validation_split=0.1)

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
def one_hot_encoder(X,total_size):
#    min_label = min(X)
#    max_label = max(X)
#    total_size = int(max_label-min_label)
    print(X[0])
    one_hot_X = np.zeros((X.shape[0],total_size))
    for i in range(X.shape[0]):
        index = int(X[i])
        one_hot_X[i,index] = 1
    return one_hot_X
"""
function " proj_plane "
Notes: defines xz as a single plane reducing dimensionality to 2
removing one output angle
"""
def create_drawers(y):
    y_class = []
    for i in range(y.shape[1]):
        drawers = (y[:,i]+90)//10
        y_class.append(one_hot_encoder(drawers,10))
#    drawers[:,1] = (y[:,1])//10
    y_class = np.array(y_class).reshape((y.shape[0],19*y.shape[1]))
    
    return y_class

def load_data():
    data = pd.read_csv('dataset.csv')
    out_cols = [col for col in data.columns if 'angle' in col]
#    print(data.describe())
    y = data[out_cols].values
    y_class = create_drawers(y)
    data.drop(out_cols,axis=1,inplace=True)
    X = data.values
    row,col = X.shape
    X_norm = X

#    X_norm = transform(X)
#    X_norm = proj_plane(X)

#    y = np.radians(y)

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
    
    return X_train, y_train, y_class_train, X_test, y_test, y_class_test
    
"""
function " main "
Notes: load dataset, transform values
train model and save
"""   
def main():
    X_train, y_train, y_class_train, X_test, y_test, y_class_test = load_data()
    
    nn = neural_net(X_train,y_class_train,False)
    nn.save('model.hd5')
#    y_pred = nn.predict(X_test)
#    print(np.degrees(y_pred))
#    print()
#    print(np.degrees(y_test))
#    print()
#    res = pd.DataFrame(np.degrees(y_pred - y_test))
#    print(res.describe())
#    print(np.degrees(np.mean(abs(y_pred-y_test),axis=0)))

def upack_drawers(y_class):
    y = []
    for i in range(4):
        value = np.where(y_class[i*19:(i+1)*19] == 1)
        y.append(value[0]*10 - 90)
    return y

def testing():
    X_train, y_train, y_class_train, X_test, y_test, y_class_test = load_data()
#    nn = load_model('model.hd5')
#    y_pred = nn.predict(X_test[:1,:])
#    y_pred[y_pred>=0.5] = 1
#    y_pred[y_pred<0.5] = 0
#    print(upack_drawers(y_pred))
    print(y_test[0])
    print(y_class_test[0])
    print(upack_drawers(y_class_test[0]))
    
if __name__ == '__main__':
#    main()
    testing()