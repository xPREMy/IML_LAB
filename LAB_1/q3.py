import numpy as np
import pandas as pd

# Load the dataset from the text file
df=pd.read_csv('iris.txt')
print(df)

# Prepare the data by dropping the non-numeric 'Iris plant' column
X=df.drop("Iris plant", axis=1).values
n, d = X.shape

# (a) Calculate the covariance matrix for the features
C = np.cov(X, rowvar=False)
print("(a) Covariance Matrix:\n", C)

# (b) Find the most and least correlated feature pairs
corr_matrix = np.corrcoef(X, rowvar=False)
triu_indices = np.triu_indices(d, k=1) # Get upper triangle indices to avoid duplicates
corr_values = corr_matrix[triu_indices]
print(corr_matrix)
max_corr_idx = np.argmax(corr_values)
min_corr_idx = np.argmin(corr_values)
feature_names = df.columns
pairs = list(zip(triu_indices[0], triu_indices[1]))

print("\n(b) Most positively correlated features:",
      (feature_names[pairs[max_corr_idx][0]], feature_names[pairs[max_corr_idx][1]]),
      "with correlation =", corr_values[max_corr_idx])

print("Most negatively correlated features:",
      (feature_names[pairs[min_corr_idx][0]], feature_names[pairs[min_corr_idx][1]]),
      "with correlation =", corr_values[min_corr_idx])

print("Least correlated (closest to 0):",
      (feature_names[pairs[np.argmin(np.abs(corr_values))][0]],
       feature_names[pairs[np.argmin(np.abs(corr_values))][1]]),
      "with correlation =", corr_values[np.argmin(np.abs(corr_values))])

# (c) Calculate the sample mean and variance for each feature
df2=df.drop('Iris plant',axis=1)
means =df2.mean()
variances = df2.var(ddof=1)
print("\n(c) Sample means:\n", means)
print("Sample variances:\n", variances)

# (d) Find the eigenvalues of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(C)
print("\n(d) Eigenvalues of covariance matrix:\n", eigenvalues)