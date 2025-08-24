# Importing numpy and math
import numpy as np 
import math

def generate_A(mean,scale,size):
  return np.random.normal(mean,scale,size) # getting an random matrix from gausian distribution

def froba_normal(A):   # defining function for frobanious normalization
  total=0
  for i in range(A.shape[0]): # loop over rows
    for j in range(A.shape[1]): # loop over columns
      total +=A[i][j]**2
  return math.sqrt(total)  # taking square root of summation of square of all elements in matrix

mean =0 
scale =1
while(True):
  m,n=int(input("m : ")),int(input("n : "))  # getting user input for size of matrix
  if m>6 and n > 8:              # conditioning for m and n     
    break
  else:
    print("Both m must be > 6 and n must be > 8. Try again.")
size=(m,n)
A=generate_A(mean,scale,size) # generating matrix
print(froba_normal(A)) # calling function to print norm