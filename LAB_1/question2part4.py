import numpy as np
import math
from question2part1 import generate_A
from question2part2 import EVD
def VTV_I(egnV):
    I=np.identity(egnV.shape[1])
    return float(np.linalg.norm(egnV.T @ egnV - I,'fro'))
if __name__ == "__main__":
    size=(100,100)
    mean=0
    sigma=math.sqrt(float(input("enter varience : ")))
    A=generate_A(size,sigma,mean)
    B=A+A.T
    egnval,egnV=EVD(B)
    print(VTV_I(egnV=egnV))