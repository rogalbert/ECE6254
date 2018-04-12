# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 16:50:19 2018

@author: Spectre
"""

# Importing Keras Sequential Model
#from keras.models import Sequential
#from keras.layers import Dense
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import h5py

# Initializing the seed value to a integer.
seed = 7

np.random.seed(seed)

# Loading the data set (PIMA Diabetes Dataset)
for file in os.listdir("Data"):
    data=pd.read_csv(file)
    X=data[['Time','Open','High','Low','Close','Volume ']]
    X['Time']=pd.to_numeric(pd.to_datetime(X['Time']))
    X=((X-np.mean(X))/np.std(X))
    y=(data['Close']/data['Close'].shift(1)-1)
    break

for col in ['Open','High','Low','Close','Volume ']:
    for i in range(1,500):
        X[col+"_"+str(i)]=X[col].shift(i)
        if i%5==0:
            X[col+"_"+'Mom_'+str(i)]=np.log(X[col].shift(1)/X[col].shift(i))
            X[col+"_"+'Rev_'+str(i)]=np.log(X[col].shift(i)/X[col].shift(1))
            X[col+'Std_'+str(i)]=X[col].rolling(i).std()
    del X[col]
    
hf = h5py.File('features.h5', 'w')
hf.create_dataset('f1_2803', data=X)
hf.close()
    
"""
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,y,test_size=0.1)
ytrain=np.where(Ytrain>0,1,0)

X=Xtrain
Y=ytrain
# Initializing the Sequential model from KERAS.
model = Sequential()

# Creating a 16 neuron hidden layer with Linear Rectified activation function.
model.add(Dense(16, input_dim=6, init='uniform', activation='relu'))

# Creating a 6 neuron hidden layer.
model.add(Dense(6, init='uniform', activation='relu'))

# Adding a output layer.
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compiling the model
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
# Fitting the model
model.fit(X, Y, nb_epoch=50, batch_size=10)

scores = model.evaluate(X, Y)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
"""

