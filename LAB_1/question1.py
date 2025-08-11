import numpy as np
import math
def Forbenius_norm_by_numpy(A):
    return math.sqrt(float(np.sum(np.square(A))))

def Forbenius_norm_by_for_loop(A):
    total=0
    (m,n)=A.shape
    for i in range(m):
        for j in range(n):
            total+=A[i][j]**2
    return math.sqrt(total)
if __name__ == "__main__":
    m,n=int(input("m: ")),int(input("n :"))
    A=np.random.normal(loc=0.0,scale=1.0,size=(m,n))
    print(A)
    print(Forbenius_norm_by_for_loop(A),Forbenius_norm_by_numpy(A))