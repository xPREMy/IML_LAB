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
            # Optional: print cluster sizes
            for k in range(self.no_of_partitions):
                print(f"Cluster {k+1} size: {len(self.partitions[k])}")
            print("-" * 40)
            
    def _2Dplot_for_two_partitions(self,X):
        plt.scatter(X[self.partitions[0],0],X[self.partitions[0],1],c='r')
        plt.scatter(X[self.partitions[1],0],X[self.partitions[1],1],c='g')
        plt.show()
    
    def _2Dplot_for_two_partitions_of_H(self):
        plt.scatter(self.x[self.partitions[0],0],self.x[self.partitions[0],1],c='r')
        plt.scatter(self.x[self.partitions[1],0],self.x[self.partitions[1],1],c='g')
        plt.xlabel("Spectral Dimension 1")
        plt.ylabel("Spectral Dimension 2")
        plt.title("after training Spectral Embeddings (2 clusters)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def performance(self,Y):
        n,d=self.x.shape
        correctc1=np.sum(Y[self.partitions[0]]==1)
        correctc2=np.sum(Y[self.partitions[1]]==2)
        b=(correctc1+correctc2)/n
        a=1-b
        return (a+b+abs(a-b))/2

# %%
class SpectralClustering:
    def __init__(self):
        self.W = None
        self.D = None
        self.L = None
        self.H = None
        self.partitions = None

    def adjacency_matrix(self, X, sigma):
        n = X.shape[0]
        self.W = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist2 = np.linalg.norm(X[i] - X[j])**2
                val = np.exp(-dist2 / (2 * sigma**2))
                self.W[i, j] = self.W[j, i] = val

    def degree_matrix(self):
        diag = np.sum(self.W, axis=1)
        self.D = np.diag(diag)

    def fit(self, X, sigma, p,Y,updates):
        ### PART A ###
        self.adjacency_matrix(X, sigma)
        self.degree_matrix()
        self.L = self.D - self.W

        eigvals, eigvecs = np.linalg.eigh(self.L)
        idx=np.argsort(eigvals)
        eigvecs=eigvecs[:,idx]
        self.H = eigvecs[:,:2]
        print(self.H.shape)
        ### PART B ####

        # ploting H
        Y = np.array(Y).flatten()
        plt.scatter(self.H[:, 0], self.H[:, 1],c='b')

        plt.xlabel("Spectral Dimension 1")
        plt.ylabel("Spectral Dimension 2")
        plt.title("Spectral Embeddings (2 clusters)")
        plt.legend()
        plt.grid(True)
        plt.show()


        ### part C ###
        model = k_means()
        self.partitions = model.fit(self.H,2,updates)
        self.partitions=model.partitions

        ### part D ###
        print(model.performance(Y)*100,"%")
        model._2Dplot_for_two_partitions_of_H()
        model._2Dplot_for_two_partitions(X)
        print("Training done.")


# %%
new_model=SpectralClustering()
new_model.fit(X,1,2,Y,40)

# %%


# %%



