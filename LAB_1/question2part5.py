import numpy as np
import math
from question2part1 import generate_A
from question2part2 import B_B_k_norm
import matplotlib.pyplot as plt
def plot_data_for_B(k,size,mean):
    bbknorm=[]
    for i in range(101):
        A=generate_A(size,math.sqrt(i*0.01),mean)
        B=A+A.T
        bbknorm.append(B_B_k_norm(B,k))
    return bbknorm
if __name__ == "__main__":
    k=[5, 10, 20, 25, 30]
    size=(100,100)
    mean=0
    BigB=[]
    varience=[]
    for i in range(101):
        varience.append(i*0.01)
    for i in k:
        BigB.append(plot_data_for_B(i,size,mean))
    # for i in range(5):
    #     plt.subplot(5,1,i+1)
    #     plt.plot(varience,BigB[i])
    #     plt.title(f"k={5*(i+1)}")
    for i in range(5):
        plt.plot(varience,BigB[i],label=f"k=: {k[i]}")
    plt.legend()
    plt.show()