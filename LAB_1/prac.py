import numpy as np
V=np.array([[1,3,5],[3,2,6],[1,7,3]])
D=np.array([[1,0,0],[0,2,0],[0,0,3]])

B=V @ D @ V.T
print(B)
egnval,egnV=np.linalg.eig(B)
print(np.argsort(egnval)[::-1])