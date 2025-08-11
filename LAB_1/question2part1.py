import numpy as np
import math
def generate_A(size,sigma,mean):
    return np.random.normal(loc=mean,scale=sigma,size=size)

def EVD(B):
    egnval,egnV=np.linalg.eigh(B)
    ind_dec=np.argsort(egnval)[::-1]
    egnval_dec=egnval[ind_dec]
    egnV_dec=egnV[:,ind_dec]
    return egnval_dec,egnV_dec

def diagonal(egnval):
    return np.diag(egnval)

if __name__ == "__main__":
    size=(100,100)
    mean=0
    sigma=math.sqrt(float(input("enter varience : ")))

    A=generate_A(size,sigma,mean)

    B=A+A.T
    print(B)

    egnval , egnV = EVD(B) 

    B_reconstructed=egnV @ diagonal(egnval=egnval) @ egnV.T

    print(B_reconstructed)


