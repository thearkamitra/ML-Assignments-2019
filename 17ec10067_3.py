
'''
#Roll : 17EC10067


#Name : Arka Mitra

#Assignment Number : 3

#Uses python3



'''

import numpy as np
import pandas as pd
import copy

data=pd.read_csv('data3_19.csv')
data_pred=pd.read_csv('test3_19.csv',header=None)


data=data.values
data_pred=data_pred.values
num_rows,m=data.shape
max_depth=10
find_pred=False



dictionary={'1st':0,'2nd':0,"3rd":0,"crew":0,"adult":1,"child":1,"female":2,"male":2}

#Just calculates the entropy given positive and negatives
def findvalue(pos,neg):
    if(pos==0 or neg==0):
        return 0
    ans=-(pos/(pos+neg))*np.log(pos/(pos+neg))-(neg/(neg+pos))*np.log(neg/(neg+pos))
    ans=ans/np.log(2)
    return ans



#Class for the node
class Createnode():
    def __init__(self,name,depth,parents=[]):
        self.name=name
        self.prob=0
        self.child=[]
        self.depth=depth
        self.parents=parents


#Calculates the whole entropy of the dataset
def entropy(datanow):
    datanow=np.asarray(datanow)
    datanow.reshape(-1,m)
    pos=0
    neg=0
    for i in range(len(datanow)):
        if datanow[i][-1]=='yes':
            pos+=1
        else:
            neg+=1
    return pos,neg,findvalue(pos,neg)


#checks if all the elements are same
def aresame(datanow):
    
    datanow=np.asarray(datanow)
    datanow.reshape(-1,m)
    for i in range(len(datanow)):
        if not(np.array_equal(datanow[0][:-1],datanow[i][:-1])):
            return False
    return True
        


#finds individual entropies to find max information gain
def individual_entropy(datanow,col):
    values={}
    
    datanow=np.asarray(datanow)
    datanow.reshape(-1,m)
    for i in range(len(datanow)):
        if datanow[i][col] in values:
            if datanow[i][-1]=='yes':
                values[datanow[i][col]][0]+=1
            else:
                values[datanow[i][col]][1]+=1
        else:
            values[datanow[i][col]]=[0,0]
    sumnow=0
    for i in values:
        
        pos=values[i][0]
        neg=values[i][1]
        sumnow+=(pos+neg)*findvalue(pos,neg)/len(datanow)
    return sumnow

##Recursively builds the tree
def choosenode(datanow,node):
    if(node.depth==max_depth):
        return
    datanow=np.asarray(datanow)
    datanow.reshape(-1,m)
    
    pos,neg,ext=entropy(datanow)
    if aresame(datanow):
        node.prob=pos*1.0/(pos+neg)
        return
    else:
        maxpos=0
        p=(node.parents).copy()
        maximum=-1000
        flag=1
        for i in range(len(datanow[0])-1):
            if i in p:
                continue
            if(maximum<(ext-individual_entropy(datanow,i))):
                flag=0
                maximum=ext-individual_entropy(datanow,i)
                maxpos=i
        if(flag):
            node.prob=pos*1.0/(pos+neg)
            return
        nodes={}
        p.append(maxpos)
        for i in range(len(datanow)):
            if datanow[i][maxpos] in nodes:
                nodes[datanow[i][maxpos]].append(datanow[i])
            else:
                node.child.append(Createnode(datanow[i][maxpos],node.depth+1,p))
                nodes[datanow[i][maxpos]]=[datanow[i]]
        for i in range(len(node.child)):
#             print(len(nodes[node.child[i].name]))
#             print(node.child[i].name)
            choosenode(nodes[node.child[i].name],node.child[i])


#predicts the individual value for a given class
def pred_indi(data_i,node):
    if len(node.child)==0 or node.depth==max_depth:
        if node.prob>=0.5:
            return "yes"
        else:
            return "no"
    for i in node.child:
        for j in data_i:
            if i.name==j:
                return pred_indi(data_i,i)
    return "no"

#predicts the values and returns the accuracy and predictions
def predict(data,node):
    pred_values=[]
    for i in range(len(data)):
        pred_values.append(pred_indi(data[i],node))
#         node2=copy.copy(node)
#         while(len(node2.child)):
#             if (node2.depth==max_depth-1):
#                 break
#             for j in range(len(node2.child)):
#                 if data[i][dictionary[node2.child[0].name]]==node2.child[j].name:
#                     node2=copy.copy(node2.child[j])
#                     break
#         if(node2.prob>=0.5):
#             pred_values.append("yes")
#         else:
#             pred_values.append("no")
    corr=0
    for i in range(len(data)):
        if pred_values[i]==data[i][-1]:
            corr+=1
    acc=corr/len(data)
    return np.array(pred_values,dtype=object),acc

#initializes the samples
sample_weights=np.ones((len(data)))/(len(data))
#Runs for the first time
M=3
nodes=[]
significances=[]
data2=data
node=Createnode("Root",0,[])
nodes.append(node)
sample_weights=np.ones((len(data)))/(len(data))
choosenode(data2,node)
y_pred,acc=predict(data2,node)

error=0
for i in range(len(data)):
    error+=sample_weights[i]*(y_pred[i]!=data2[i][-1])
significance=0.5*np.log((1-error)/(error))
significances.append(significance)
sample_weights*=np.exp(significance*(1-2*(y_pred!=data2[:,-1])))
sample_weights=sample_weights/np.sum(sample_weights)
   
#Runs for the rest epochs
for i in range(M-1):
    
    node=Createnode("Root",0,[])
    nodes.append(node)
    arr=np.random.choice(np.arange(len(data)),len(data),replace=True,p=sample_weights)
    data3=data2[arr]
    data2=data3
    a=sample_weights[arr]
    sample_weights=a
    sample_weights=np.ones((len(data)))/(len(data))
    choosenode(data2,node)
    y_pred,acc=predict(data2,node)
    error=0
    for j in range(len(data)):
        error+=sample_weights[j]*(y_pred[j]!=data2[j][-1])
    # print(error)
    significance=0.5*np.log((1-error)/(error))
    significances.append(significance)
    sample_weights*=np.exp(significance*(1-2*(y_pred!=data2[:,-1])))
    sample_weights=sample_weights/np.sum(sample_weights)
#Predicts the values on the adaboost values
def predict_test(data):
    predictions=np.zeros(len(data),dtype=float)
    for i in range(len(significances)):
        ypred,acc=predict(data,nodes[i])
        for j in range(len(ypred)):
            if ypred[j]=='yes':
                ypred[j]=1*significances[i]
            else:
                ypred[j]=-1*significances[i]
        ypred=np.array(ypred,dtype=np.float)
        predictions+=ypred
    corr=0
    for i in range(len(predictions)):
        if predictions[i]>=0:
            temp='yes'
        else:
            temp='no'
        if temp==data[i][-1]:
            corr+=1
    acc=corr/len(data)
    
    return acc
#Prints the tree
def print_Tree(node):
    for i in node.child:
        for _ in range(node.depth):
            print("  ",end="")
        if(len(i.child)):
            print(i.name)
        else:
            print(i.name+"  ",end="")#+str(i.prob),end=" ")
            if(i.prob>=0.5):
                print('Yes')
            else:
                print('No')
        print_Tree(i)

print("The test accuracy is "+str(predict_test(data_pred)*100)+" %")