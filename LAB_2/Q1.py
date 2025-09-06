# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Load dataset and shuffle rows
df=pd.read_excel("Q1.xlsx")
df=df.sample(frac=1,random_state=75).reset_index(drop=True)
print(df)

# Load dataset and shuffle rows
X=df.drop("Y",axis=1).values.astype(float)
Y=df["Y"].values.astype(float)
Y=np.expand_dims(Y,axis=1) # reshape Y to column vector
print(X)

# %%
# Scatter plots of each feature vs target
for i in range(8):
    plt.scatter(df.iloc[:,i].values,df["Y"].values)
plt.show()

# %%
# Print feature-wise min and max
min=np.min(X,axis=0)
print(min)
max=np.max(X,axis=0)
print(max)

# %%
# Custom Min-Max Scaler class
class MinMaxScaler:
    def __init__(self):
        self.mins=None
        self.maxs=None
    def fit_transform(self,X):
        min=np.min(X,axis=0)
        self.mins=min
        print(min)
        max=np.max(X,axis=0)
        self.maxs=max
        print(max)
        return (X-min)/(max-min)
    def transform(self,x):
        return (x-self.mins)/(self.maxs-self.mins)
    
# Apply scaling
scaler = MinMaxScaler()  
X=scaler.fit_transform(X)
min=np.min(X,axis=0)
print(min)
# Scatter plots after scaling
for i in range(8):
    plt.scatter(X[:,i],Y)
plt.show()
ones=np.ones((X.shape[0],1))
X=np.column_stack((X,ones))
print(X)

# %%
def train_test_split(X,Y,split):
    # split belongs [0,1]
    split_num=int(Y.shape[0]*split)
    Y_1=Y[:split_num,:]
    Y_2=Y[split_num:,:]
    X_1=X[:split_num,:]
    X_2=X[split_num:,:]
    return X_1,X_2,Y_1,Y_2
# seperating traing set for 70%
x_train,xtem,y_train,ytem=train_test_split(X,Y,split=0.7)
print(x_train.shape,y_train.shape)
# seperating cross validation(15%) and testing set(15%) split=0.5
x_val,x_test,y_val,y_test=train_test_split(xtem,ytem,split=0.5)
print(x_test.shape,y_test.shape)

# %%
theta=np.zeros((X.shape[1],1)) # defined theta for parameters

# Hypothesis function
def h(xi,theta):
    xi=np.expand_dims(xi,axis=0)
    # xi will be an matrix so x_i.shape == (1,9)
    r=xi@theta
    return r[0][0]

# Mean Squared Error
def error(X,theta,Y):
    r=((X@theta-Y).T @ (X@theta-Y))/X.shape[0]
    return r[0][0]

print(h(x_train[0],theta),error(X,theta,Y))

# %% [markdown]
# Q1 part 1

# %%
XtX = x_train.T @ x_train
print("cond(X^T X) =", np.linalg.cond(XtX))
# conditioning value is much high so we will use np.linalg.pinv(X.T@X)it will give an pseudo matrix

# %%
# Closed-form solution (Normal Equation using pseudo-inverse)
theta_1 = np.linalg.pinv(x_train.T @ x_train) @ (x_train.T @ y_train)
print(theta_1,theta_1.shape)
err=error(x_train,theta_1,y_train)
print("training error",err)
print("validation error :",error(x_val,theta_1,y_val))
print("testing error",error(x_test,theta_1,y_test))


# %% [markdown]
# Q1 part 2

# %%
# Linear Regression using Gradient Descent
class linear_regression_model:
    def __init__(self):
        self.weights=None
        self.alpha=None
        self.error_T=[]
        self.error_val=[]
        self.epochs=None
    def fit(self,xt,yt,xv,yv,alpha,epochs):
        self.weights=np.zeros((xt.shape[1],1)) # weight initialization
        self.alpha=alpha
        self.epochs=epochs
        for i in range(self.epochs):
            m=xt.shape[0]
            # Gradient descent update rule
            self.weights-=(self.alpha*2/(m))*(xt.T@(xt@self.weights-yt))
            # Track training and validation error
            train_err=error(xt,self.weights,yt)
            val_err=error(xv,self.weights,yv)
            self.error_T.append(train_err)
            self.error_val.append(val_err)
            print(f"Epoch {i+1}/{self.epochs} | Train Error: {train_err:.4f} | Val Error: {val_err:.4f}")
    def training_history(self):
        return self.error_T
    def validation_history(self):
        return self.error_val
    def weightsa(self):
        return self.weights
    def prediction(self,x):
            return x@self.weights
    def prediction_for_without_scaled(self,x):
            x=scaler.transform(x)
            return x@self.weights
    def errorplot(self):
        epoch=np.arange(1,self.epochs+1)
        plt.plot(epoch,self.error_T,label='training error')
        plt.plot(epoch,self.error_val,label='validation error')
        plt.xlabel("epochs")
        plt.ylabel("error")
        plt.title("error vs epochs")
        plt.legend()
        plt.show()

# %%
# Train model with Gradient Descent
linear_regression=linear_regression_model()
linear_regression.fit(x_train,y_train ,x_val ,  y_val , alpha=0.31,epochs=30000)

# %%
# Compare weights from closed-form vs gradient descent
linear_regression.errorplot()
print("theta1 : \n",theta_1)
theta_2=linear_regression.weights
print("theta2 : \n",theta_2)


# %%
# Final errors
training_error=error(x_train,theta_2,y_train)
validation_error=error(x_val,theta_2,y_val)
testing_error=error(x_test,theta_2,y_test)
print("training error :",training_error)
print("validation error :",validation_error)
print("testing error :",testing_error)

# %%



