# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# %%
df=pd.read_csv("dataset.txt",header=None,sep="\s+")
X=df.drop(df.columns[2],axis=1).values.astype(float)
Y=df[df.columns[2]].values.astype(float).reshape(-1,1)
print(X,type(X),X.shape,Y.shape)

# %%
plt.scatter(X[:,0].reshape(-1,1),X[:,1].reshape(-1,1))
plt.show()

# %%
class k_means:
    def __init__(self):
        self.partitions=None
        self.partition_means=None
        self.no_of_partitions=None
        self.x=None
    
    ### PART A ###  
    def random_initialization(self):
        self.partition_means=[]
        n,d=self.x.shape
        for i in range(self.no_of_partitions):
            mins=np.min(self.x,axis=0)
            maxs=np.max(self.x,axis=0)
            mean_i=[]
            for j in range(d):
                random_num=random.uniform(mins[j],maxs[j])
                mean_i.append(random_num)
            self.partition_means.append(mean_i)
        self.partition_means=np.array(self.partition_means)

    def partition_update(self):
        n,d=self.x.shape
        self.partitions=[[] for _ in range(self.no_of_partitions)]
        partitions_distances= np.linalg.norm(
        self.x[:, np.newaxis, :] - self.partition_means[np.newaxis, :, :], axis=2
        )
        for i in range(n):
            arg=np.argmin(partitions_distances[i])
            self.partitions[arg].append(i)
        for i in range(self.no_of_partitions):
            self.partition_means[i]=np.mean(self.x[self.partitions[i]],axis=0)

    def fit(self,x,p,updates):
        self.x=x
        self.no_of_partitions=p
        n,d=x.shape
        self.partitions = [[] for _ in range(self.no_of_partitions)]
        self.random_initialization()
        for i in range(updates):
            self.partition_update()
            print(f"Epoch {i+1}/{updates}")
            for k in range(self.no_of_partitions):
                print(f"Cluster {k+1} size: {len(self.partitions[k])}")
            print("-" * 40)
    
    ### PART B #### 
    def _2Dplot_for_two_partitions(self):
        plt.scatter(self.x[self.partitions[0],0],self.x[self.partitions[0],1],c='r')
        plt.scatter(self.x[self.partitions[1],0],self.x[self.partitions[1],1],c='g')
        plt.xlabel("dimension1")
        plt.ylabel("dimension 2")
        plt.show()

    ### part C ###
    def performance(self):
        n,d=self.x.shape
        correctc1=np.sum(Y[self.partitions[0]]==2)
        correctc2=np.sum(Y[self.partitions[1]]==1)
        b=(correctc1+correctc2)/n
        a=1-b
        return (a+b+abs(a-b))/2

# %%
model=k_means()
model.fit(X,2,9)

# %%
model._2Dplot_for_two_partitions()
print(model.performance())

# %%
