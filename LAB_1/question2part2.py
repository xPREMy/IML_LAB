import numpy as np
import math
from question2part1 import EVD , generate_A
def B_B_k_norm(B,k):
    egnval,egnV=EVD(B)
    B_k=np.zeros(B.shape)
    for i in range(k):
        v_i=np.expand_dims(egnV[:,i],axis=1)
        B_k+=egnval[i]*( v_i @ v_i.T )
    return float(np.linalg.norm(B-B_k,'fro'))
if __name__ == "__main__":
    size=(100,100)
    mean=0
    sigma=math.sqrt(float(input("enter varience : ")))
    A=generate_A(size,sigma,mean)
    B=A+A.T
    k=int(input("enter value of K : "))
    print(B_B_k_norm(B,k))