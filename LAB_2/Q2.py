# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
# Load dataset
df= pd.read_csv("Q2.csv",header=None)
df=df.sample(frac=1,random_state=30).reset_index(drop=True)# shuffle data
df.columns=["Y","x1","x2"]

# Prepare labels (Y) and features (X)
Y=df["Y"].values.astype(float)
Y=np.expand_dims(Y,axis=1)
Y=np.where(Y==-1,0,Y) # map -1 to 0 for logistic regression
X=df.drop("Y",axis=1).values.astype(float) 
ones=np.ones((X.shape[0],1)) # adding intercept
X=np.column_stack((X,ones))
print(Y,X.shape)

# %%
# Scatter plot of dataset with color mapping
clr=df["Y"].map({1:"orange",-1:"blue"})
plt.scatter(df["x1"],df["x2"],c=clr)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

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
print(x_val.shape,y_val.shape)
print(x_test.shape,y_test.shape)

# %%
# Hypothesis function (sigmoid)
theta=np.zeros((x_train.shape[1],1))
def H(x,theta):
    theta=theta.reshape(-1,1)
    return 1/(1+np.exp(-(x@theta)))
print(H(x_train,theta),H(x_train,theta).shape)

# %%
def loss(X,Y,theta):
    m=X.shape[0]
    eps = 1e-15
    return -(1/m)*np.sum((Y)*np.log(H(X,theta)+eps)+(1-Y)*np.log(1-(H(X,theta)-eps)))
print(loss(x_train,y_train,theta))

# %%
# Logistic Regression Class (gradient descent implementation)
class logistic_regression():
    def __init__(self):
        self.weights=None
        self.training_loss=[]
        self.validation_loss=[]
        self.epochs=None
        self.lr=None
        self.confusion_matrix=None
        self.training_accuracy=[]
        self.validation_accuracy=[]
    
    def accuracy_score(self,x,y,threshold):
        total=x.shape[0]
        xpred=H(x,self.weights)
        xpred=(xpred>=threshold).astype(int).ravel()
        y=y.astype(int).ravel()
        correct=np.sum(y==xpred)
        return correct/total

    def fit(self,xt,yt,xv,yv,lr,epochs):
        self.weights=np.zeros((xt.shape[1],1))
        self.epochs=epochs
        self.lr=lr
        for i in range(self.epochs):
            m=xt.shape[0]
            self.weights-=(self.lr/(m))*(xt.T@(H(xt,self.weights)-yt))

            # Compute training & validation loss
            train_loss=loss(xt,yt,self.weights)
            val_loss=loss(xv,yv,self.weights)
            self.training_loss.append(train_loss)
            self.validation_loss.append(val_loss)

            # Compute accuracy
            self.training_accuracy.append(self.accuracy_score(xt,yt,0.5))
            self.validation_accuracy.append(self.accuracy_score(xv,yv,threshold=0.5))
            print(f"Epoch {i+1}/{self.epochs} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
            
    def confusion_matrixfn(self,x,y,threshold,show):
        xpred=H(x,self.weights)
        xpred=(xpred>=threshold).astype(int).ravel()
        y=y.astype(int).ravel()
        m=np.unique(y)
        confusion_matrix=[]
        # Confusion matrix calculation
        for a in m:
            pred_row=[]
            for b in m:
                mask_1=(y==b)
                mask_2=(xpred==a)
                count=np.sum(mask_1 & mask_2)
                pred_row.append(int(count))
            confusion_matrix.append(pred_row)
        self.confusion_matrix=confusion_matrix
        if show:
            df=pd.DataFrame(confusion_matrix,index=[f"PRED {cls}" for cls in m],columns=[f"TRUE {cls}" for cls in m])
            print(df)

    # Precision = TP / (TP + FP)
    def precision_for_binary_classification(self):
        if self.confusion_matrix[1][1]+self.confusion_matrix[1][0] == 0:
            return float(0)
        return self.confusion_matrix[1][1]/(self.confusion_matrix[1][1]+self.confusion_matrix[1][0])
    
    # Recall = TP / (TP + FN)
    def recall_for_binary_classification(self):
        if self.confusion_matrix[1][1]+self.confusion_matrix[0][1] == 0:
            return float(0)
        return self.confusion_matrix[1][1]/(self.confusion_matrix[1][1]+self.confusion_matrix[0][1])
       
    def lossplot(self):
        epoch=np.arange(1,self.epochs+1)
        plt.plot(epoch,self.training_loss,label='training loss')
        plt.plot(epoch,self.validation_loss,label='validation loss')
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("Training and Validation loass vs. Iterations")
        plt.legend()
        plt.show()
    
    def plot_training_validation_accuracy(self):
        epochs=np.arange(1,self.epochs+1)
        self.training_accuracy=np.array(self.training_accuracy)
        self.validation_accuracy=np.array(self.validation_accuracy)
        plt.plot(epochs,self.training_accuracy,label="training")
        plt.plot(epochs,self.validation_accuracy,label="validation")
        plt.title("Training and Validation accuracy vs. Iterations")
        plt.legend()
        plt.show()

# %%
# Instantiate and train logistic regression model
logistic_model=logistic_regression()
logistic_model.fit(x_train,y_train,x_val,y_val,0.1,7300)

# %%
# Plot loss and accuracy over iterations
logistic_model.lossplot()
logistic_model.plot_training_validation_accuracy()

# %%
# Print accuracies and confusion matrices for train, validation, test sets
print("training accuracy :",logistic_model.accuracy_score(x_train,y_train,0.5))
print("validation accuracy",logistic_model.accuracy_score(x_val,y_val,0.5))
print("testing accuracy",logistic_model.accuracy_score(x_test,y_test,0.5))
print("training set confusion matrix\n")
logistic_model.confusion_matrixfn(x_train,y_train,0.5,show=True)
print("\n\nvalidation set confusion matrix\n")
logistic_model.confusion_matrixfn(x_val,y_val,0.5,show=True)
print("\n\ntesting set confusion matrix\n")
logistic_model.confusion_matrixfn(x_test,y_test,0.5,show=True)

# %%
def F1_score(X,Y,threshold):
    logistic_model.confusion_matrixfn(X,Y,threshold,show=False)
    precision=logistic_model.precision_for_binary_classification()
    print("precision : ", precision)
    recall=logistic_model.recall_for_binary_classification()
    print("recall",recall)
    return (2*precision*recall)/(precision+recall)
print("\n\ntraining set")
print("F1 SCORE: ",F1_score(x_train,y_train,0.5))
print("\n\nvalidation set")
print("F1 SCORE: ",F1_score(x_val,y_val,0.5))
print("\n\ntesting set")
print("F1 SCORE: ",F1_score(x_test,y_test,0.5))

# %%
precision_plot=[]
recall_plot=[]
thresholds=np.arange(0,1,0.05)
# training plot
for i in thresholds:
    logistic_model.confusion_matrixfn(x_train,y_train,i,show=False)
    precision=logistic_model.precision_for_binary_classification()
    recall=logistic_model.recall_for_binary_classification()
    precision_plot.append(precision)
    recall_plot.append(recall)
plt.scatter(recall_plot,precision_plot)
plt.xlabel("recall")
plt.ylabel("precision")
plt.title("precision VS recall for TRAINING SET")
plt.show()

# %%
precision_plot=[]
recall_plot=[]
# validation plot
for i in thresholds:
    logistic_model.confusion_matrixfn(x_val,y_val,i,show=False)
    precision=logistic_model.precision_for_binary_classification()
    recall=logistic_model.recall_for_binary_classification()
    precision_plot.append(precision)
    recall_plot.append(recall)
plt.scatter(recall_plot,precision_plot)
plt.xlabel("recall")
plt.ylabel("precision")
plt.title("precision VS recall for VALIDATION SET")
plt.show()

# %%
precision_plot=[]
recall_plot=[]
# testing plot
for i in thresholds:
    logistic_model.confusion_matrixfn(x_test,y_test,i,show=False)
    precision=logistic_model.precision_for_binary_classification()
    recall=logistic_model.recall_for_binary_classification()
    precision_plot.append(precision)
    recall_plot.append(recall)
plt.scatter(recall_plot,precision_plot)
plt.xlabel("recall")
plt.ylabel("precision")
plt.title("precision VS recall for VALIDATION SET")
plt.show()

# %%
weights=logistic_model.weights
plt.scatter(df[df["Y"] == 1]["x1"],
            df[df["Y"] == 1]["x2"],
            c="orange",
            label="Class 1") 
plt.scatter(df[df["Y"] == -1]["x1"],
            df[df["Y"] == -1]["x2"],
            c="blue",
            label="Class -1")
plt.plot(df["x1"],-(weights[2]+df["x1"]*weights[0])/weights[1]) # decision boundary line
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Decision boundary")
plt.legend()
plt.show()

# %%



