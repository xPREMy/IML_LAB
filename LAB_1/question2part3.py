from question2part2 import B_B_k_norm
import numpy as np
import matplotlib.pyplot as plt
import math
from question2part1 import generate_A
if __name__ == "__main__":
    size=(100,100)
    mean=0
    sigma=math.sqrt(float(input("enter varience : ")))
    A=generate_A(size,sigma,mean)
    B=A+A.T
    For=[]
    K=[]
    for i in range(100):
        K.append(i+1)
        For.append(B_B_k_norm(B,i))
    plt.plot(K,For)
    plt.xlabel("k")
    plt.ylabel("normalization")
    plt.show()