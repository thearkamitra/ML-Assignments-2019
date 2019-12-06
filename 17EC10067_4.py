
'''
#Roll : 17EC10067


#Name : Arka Mitra

#Assignment Number : 4

#Uses python3



'''

import numpy as np
import pandas as pd
import copy

data=pd.read_csv("data4_19.csv",header=None)#Reads the data

data=data.values


classes=np.unique(data[:,-1])#Get sthe unique classes

data_x=data[:,:-1]
data_y=data[:,-1]

data_x=data_x.astype(float)
data_mean=np.mean(data_x,axis=0)
data_std=np.mean(data_x,axis=0)
data_x=(data_x-data_mean)/data_std#Normalizes the data so that each parameter have about equal contribution.

centroids=data_x[np.random.choice(np.arange(len(data_x)),size=3,replace=False)]#Chooses the centroids

iterations=10

for iteration in range(iterations):
    values=np.zeros(centroids.shape)
    numbers=np.zeros(len(centroids),dtype=int)
    for data_val in data_x:
        a=np.argmin(np.sum(np.square(centroids-data_val),axis=1))
        values[a]+=data_val
        numbers[a]+=1
    for i in range(len(centroids)):
        if numbers[i]!=0:
            values[i]=values[i]/numbers[i]
    centroids=copy.deepcopy(values)#Creates a deepcopy

classcount={}
predcount=[{},{},{}]
for j in range(len(data_y)):#Creates the 
    i=data_y[j]
    a=np.argmin(np.sum(np.square(centroids-data_x[j]),axis=1))
    if i not in classcount:
        classcount[i]=1
        predcount[a][i]=1
    else:
        classcount[i]+=1
        if i not in predcount[a]:
            predcount[a][i]=0
        predcount[a][i]+=1
print("The clusters are")
print(predcount)
print("\nThe centroids are:")
print(centroids*data_std+data_mean)


def jacquard_dist(predcount,classcount):#Calculates the jacquard distance for all clusters and truth values.
    total_dist=0
    for i in range(len(predcount)):
        total=0
        val_max=-1
        pos=0
        for k in predcount[i]:
            total+=predcount[i][k]
        for j in classcount:
            if j not in predcount[i]:
                val=0
            else:
                val=predcount[i][j]/(total+classcount[j]-predcount[i][j])
            if val>val_max:
                val_max=val
                pos=j
            print("The jacquard distance between cluster "+str(i)+" and "+j+" is "+str(1-val))
        print("The cluster "+str(i)+" likely represents the class "+ pos)
        total_dist+=1-val_max
    return total_dist

print("The total jacquard distance of the likely classes are "+str(jacquard_dist(predcount,classcount)))
