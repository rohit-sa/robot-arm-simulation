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
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def custom_loss(y_true,y_pred):
    W = np.array([4,8,6,2])/20
    return K.mean(K.square(y_pred - y_true)*W, axis=-1)

def ann_test(X,y,prev_model=False):
    nn_name = Path('model.hd5')
    if prev_model & nn_name.exists():
        print('Using prev model')
        nn_architecture = load_model('model.hd5')
    else:
        nn_architecture = Sequential()
        layer_param = dict(kernel_initializer='truncated_normal',
                           activation='tanh',
                           bias_initializer='ones')
        nn_architecture.add(Dense(100, input_shape=(4,), **layer_param))
#        nn_architecture.add(Dense(15, **layer_param))
        nn_architecture.add(Dense(4,activation='linear'))
        
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        nn_architecture.compile(loss='mean_squared_error', optimizer=sgd)
    
    history = nn_architecture.fit(X,y,verbose=2,epochs=250,shuffle=True,batch_size=100,validation_split=0.1)

    loss = history.history['loss']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(len(loss)),loss)
    plt.show()
    return nn_architecture

def transform(X):
    r = np.linalg.norm(X,axis=1)
    theta = (np.pi/2) - np.arccos(X[:,1]/r)
    phi = np.arctan2(X[:,2],X[:,0])
    return np.vstack((r,theta,phi)).T 

def proj_plane(X):
    x,y,z = X[:,0], X[:,1], X[:,2]
    xz = np.sqrt(x*x + z*z)
    return np.vstack((xz,y)).T
    
def main():
    data = pd.read_csv('dataset.csv')
    out_cols = [col for col in data.columns if 'angle' in col]
    y = data[out_cols].values
    data.drop(out_cols,axis=1,inplace=True)
    X = data.values
    row,col = X.shape
    X_norm = X
    X_norm = transform(X)
#    X_norm = proj_plane(X)
    y = np.radians(y)
    o = np.sum(y,axis=1)
    X_norm = np.column_stack((X_norm,o))
    print(X_norm.shape)
    print(y.shape)
    
    train_index = np.random.choice(row,int(0.85*row),replace=False)
    test_index = np.setdiff1d(range(row),train_index,assume_unique=True)
    
    X_train = X_norm[train_index]
    y_train = y[train_index]

    X_test = X_norm[test_index]
    y_test = y[test_index]
    
    
    nn = ann_test(X_train,y_train,False)
    nn.save('model.hd5')
    y_pred = nn.predict(X_test)
    
    print(np.degrees(y_pred))
    print()
    print(np.degrees(y_test))
    print()
    res = pd.DataFrame(np.degrees(y_pred - y_test))
    print(res.describe())
    print(np.degrees(np.mean(abs(y_pred-y_test),axis=0)))


if __name__ == '__main__':
    main()