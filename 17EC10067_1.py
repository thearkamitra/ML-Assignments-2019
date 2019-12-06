'''
#Roll : 17EC10067


#Name : Arka Mitra

#Assignment Number : 1

#Uses python3
#In order to predict, put find_pred=True. The input to the predict function needs to be given manually.
#Pandas have only been used once to load the .csv file.



'''

import numpy as np
import pandas as pd


data=pd.read_csv('data1_19.csv')

data=data.values

num_rows,m=data.shape

find_pred=False

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
        
            
        



node=Createnode("Root",0,[])#Creates the root node
choosenode(data,node)#starts the process
##Prints the tree
def print_Tree(node):
    for i in node.child:
        for j in range(node.depth):
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


print_Tree(node)



if (find_pred):

	data2=pd.read_csv("pred.csv")###The data should be stored in .csv format for this case. Other ways can also be implemented
	to_pred=data2.values


	Predictions=[]


	def predict(datanow,node):
	    if len(node.child)==0:
	        if node.prob>=0.5:# The thresholding can be changed
	            return "YES"
	        return "NO"
	    for i in node.child:
	        for j in datanow:
	            if i.name==j:
	                return predict(datanow,i)
	    return "NO"


	for i in range(len(to_pred)):
	    Predictions.append(predict(to_pred[i],node))


	print(Predictions)
