
'''
#Roll : 17EC10067


#Name : Arka Mitra

#Assignment Number : 6

#Uses python3

# Extra Assignment. The data.csv and test.csv are from previous year.

# MSE was used as the loss

'''
import copy
import numpy as np
import pandas as pd

data_train=pd.read_csv("data6.csv",header=None)
data_test=pd.read_csv("test6.csv",header=None)

data_train=data_train.values
data_test=data_test.values

train_x=data_train[:,:-1]
train_y=data_train[:,-1]
train_y=train_y[:,None]

losses=[]
learning_rate=0.33



weights=np.random.random((train_x.shape[-1],1))

bias=np.random.random((1,1))
iterations=10

def predictions(data,weights,bias):
    preds=np.add(np.dot(data,weights),bias)
    return preds



def loss(ypred,train_y):
    loss= 0.5*np.mean(np.square(ypred-train_y))
    return loss

def gradients(ypred,train_y,train_x,weights,bias):
    del_weights=np.mean(np.multiply(ypred-train_y,train_x),axis=0)
    del_bias=np.mean(ypred-train_y)

    return del_weights,del_bias

for i in range(iterations):
    ypred=predictions(train_x,weights,bias)
    losses.append(loss(ypred,train_y))
    del_weights,del_bias=gradients(ypred,train_y,train_x,weights,bias)
    weights=weights-learning_rate*del_weights[:,None]
    bias=bias-learning_rate*bias




def predict(data,weights,bias,threshold=0.5):
    preds=predictions(data,weights,bias)
    L=[1 if x>=threshold else 0 for x in preds]
    return L

def writetofile(L,filetowrite="17EC10067_6.out"):
    string=""
    for x in L:
        string=string+str(x)+" "
    file=open(filetowrite,"w+")
    file.writelines(string)
    file.close()
L1=predict(train_x,weights,bias)
correct=0
for i in range(len(train_x)):
    if L1[i]>=0.5:
        if train_y[i]==1:
            correct+=1
    else:
        if train_y[i]==0:
            correct+=1
print("The number of correctly classified in 20 training sets are "+str(correct))

L=predict(data_test,weights,bias)
writetofile(L)