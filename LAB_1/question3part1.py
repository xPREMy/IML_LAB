import pandas as pd
import numpy as np
def str_to_num(iris):
    iris_plant={'Iris-setosa':1,"Iris-versicolor":2,'Iris-virginica':3}
    return iris_plant[iris]
df=pd.read_csv("iris.txt")
df['iris_num']=df["Iris plant"].apply(str_to_num)

df=df.drop("Iris plant",axis=1)
df=df.values.astype("float32")
cov_matrix=np.cov(df)
print(cov_matrix)