# -*- coding: utf-8 -*-
"""
Created on Fri May 18 11:55:12 2018

@author: Ayoub El khallioui
"""
import numpy as np
from CNN_tf_layers import CNN

mnist=np.load('D:\\python\\MNN from scratch\\mnist_scaled.npz')
files=mnist.files    
X_train,y_train,X_test,y_test=[mnist[f] for f in files] 

CNN_model=CNN(epochs=20,batch_size=60,nbr_classes=10,n_features=X_train.shape[1],learning_rate=0.01,dropout_rate=0.5,shuffle=True,random_seed=10)
CNN_model.fit(X_train,y_train)
#save the model
CNN_model.save_model(epoch=20)
del CNN_model

CNN_model2=CNN(epochs=20,batch_size=60,nbr_classes=10,n_features=X_train.shape[1],learning_rate=0.01,dropout_rate=0.5,shuffle=True,random_seed=10)
#load the model
CNN_model2.load(epoch=20,path='./model_CNN/')
prediction=CNN_model2.predict(X_test,return_proba=False)
accuracy_score=np.sum(y_test==prediction)/len(y_test)
print("accuracy score={}".format(accuracy_score))

