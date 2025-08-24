import numpy as np
def MSE(ypred,ytrain):
    m=len(ypred)
    return np.sum((ypred-ytrain)**2)/m
def BCE(ypred,ytrain):
    e=1e-8
    return -np.mean(ytrain*np.log(ypred+e)+(1-ytrain)*np.log(1-ypred+e))
def functiongrad(fn,Z):
    if fn == 'sigmoid':
        return sigmoid(Z)*(1-sigmoid(Z))
    elif fn== 'linear':
        return np.ones_like(Z)
    elif fn== 'relu':
        return (Z>0).astype(float)
    else :
        return np.zeros_like(Z)
def relu(x):
    return np.maximum(x,0).astype(float)
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return (1/(1+np.exp(-x))).astype(float)
class layer:
    def __init__(self,units,activation):
        self.units=units
        self.featuresize=0
        self.activation=activation
        self.W=None
        self.B=None
        self.Z=None
        self.Ztr=None
        self.previous=None
        self.Ztrprev=None
        self.A= None
        self.Aprev=None
    def initializewb(self,x,previous):
         self.featuresize=x.shape[0]
         self.previous=previous
         if self.activation == 'relu':
                self.W = np.random.randn(self.units, self.featuresize) * np.sqrt(2 / self.featuresize)
         elif self.activation == 'sigmoid':
                self.W = np.random.randn(self.units, self.featuresize) * np.sqrt(1 / self.featuresize)
         else:
                self.W = np.random.randn(self.units, self.featuresize)
         if self.B is None:
            self.B =np.random.randn(self.units,1)
    def forwardtrain(self,xtrain,Z):
        self.Aprev=xtrain
        self.Ztrprev=Z
        self.Ztr = xtrain @ self.W.T + self.B.T
        if self.activation == 'relu':
            self.A=relu(self.Ztr)
            return self.A ,self.Ztr
        elif self.activation == 'linear':
            self.A=self.Ztr
            return self.A ,self.Ztr
        elif self.activation == 'sigmoid':
            self.A=sigmoid(self.Ztr)
            return self.A , self.Ztr
    def forwardtest(self,xtest):
        self.Aprev=xtest
        self.Ztr = xtest @ self.W.T + self.B.T
        if self.activation == 'relu':
            self.A=relu(self.Ztr)
            return self.A 
        elif self.activation == 'linear':
            self.A=self.Ztr
            return self.A 
        elif self.activation == 'sigmoid':
            self.A=sigmoid(self.Ztr)
            return self.A 
    def forward(self,x):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        self.Z=self.W @ x + self.B
        if self.activation == 'relu':
            return relu(self.Z)
        elif self.activation == 'linear':
            return self.Z
        elif self.activation == 'sigmoid':
            return sigmoid(self.Z)
    def update(self,dz,lr):
        if self.previous != None:
            Wt=self.W
            self.W=self.W - lr*(dz.T@self.Aprev)/self.Ztr.shape[0]
            self.B=self.B - lr*(np.sum(dz.T,axis=1,keepdims=True))/self.Ztr.shape[0]
            return (dz@Wt)*functiongrad(self.previous,self.Ztrprev)
class Model:
    def __init__(self,lis):
        self.layers=[]
        self.xtrain=None
        self.ytrain=None
        self.loss=None
        self.lr=None
        self.epochs=None
        self.loss_history=[]
        for i in lis:
            self.layers.append(i)
    def initialize(self,xtrain,loss,learningrate):
        x=xtrain[0].reshape(-1, 1)
        self.loss=loss
        self.lr=learningrate
        self.layers[0].initializewb(x,None)
        x=self.layers[0].forward(x)
        for j in range(1,len(self.layers)):
            self.layers[j].initializewb(x,self.layers[j-1].activation)
            x=self.layers[j].forward(x)
    def fit(self,x,y,epochs):
        self.xtrain=x
        self.ytrain = y.reshape(-1, 1)
        self.epochs=epochs
        for i in range(self.epochs):
            print(f"epochs == ",i+1,f"/ {self.epochs}")
            ypred=x
            ztr=None
            for l in self.layers:
                ypred,ztr=l.forwardtrain(ypred,ztr)
            if self.loss == 'MSE':
                cost=MSE(ypred,self.ytrain)
                print(cost)
            if self.loss == 'BCE':
                cost=BCE(ypred,self.ytrain)
                print(cost)
            self.loss_history.append(cost)
            dz=(ypred-self.ytrain)
            for l in reversed(self.layers):
                dz=l.update(dz,self.lr)
    def predict(self,x):
        y=x
        for i in self.layers:
            y=i.forwardtest(y)
        return y
    
if __name__ == "__main__":
    pass