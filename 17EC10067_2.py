'''
#Roll : 17EC10067


#Name : Arka Mitra

#Assignment Number : 2

#Uses python3



'''
import numpy as np
#Cleaninig data
file1=open("data2_19.csv")
data=file1.readlines()
data=data[1:]
labels=[]
values=[]
for i in range(len(data)):
    temp=data[i]
    temp=temp[1:-2]
    temp=temp.split(",")
    labels.append(temp[0])
    values.append(temp[1:])
values=np.array(values,dtype=int)
labels=np.array(labels,dtype=int)
values=values.reshape(-1,6)
labels=labels.reshape(-1,1)
file1.close()
#Processing the look_up matrix
look_up=np.zeros((2,6,5),dtype=float)
for i in range(len(labels)):
    for j in range(6):
        look_up[labels[i][0]][j][values[i][j]-1]+=1
neg=np.sum(look_up,axis=2)[:,0][0]
pos=np.sum(look_up,axis=2)[:,0][1]
look_up[0]=(look_up[0]+1)/(neg+5)
look_up[1]=(look_up[1]+1)/(pos+5)
p_neg=neg/(pos+neg)
p_pos=1-p_neg





##################Testing starts############################
file2=open("test2_19.csv")#Change this to predict any other file
data=file2.readlines()
data=data[1:]
labels=[]
values=[]
for i in range(len(data)):
    temp=data[i]
    temp=temp[1:-2]
    temp=temp.split(",")
    labels.append(temp[0])
    values.append(temp[1:])
values=np.array(values,dtype=int)
labels=np.array(labels,dtype=int)
values=values.reshape(-1,6)
labels=labels.reshape(-1,1)
file2.close()
correct=0
for i in range(len(labels)):
    p=p_pos
    n=p_neg
    for j in range(6):
        p=p*look_up[1][j][values[i][j]-1]
        n=n*look_up[0][j][values[i][j]-1]
    if(p>=n):
        answer=1
    else:
        answer=0
    if(answer==labels[i][0]):
        correct+=1

print("The accuracy is "+str(correct*100.0/len(labels)))