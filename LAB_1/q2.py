# Importing necessary libraries
import numpy as np 
import math
import matplotlib.pyplot as plt

# --- Function Definitions ---

def generate_A(mean,scale,size):
  """Generates a random matrix from a Gaussian distribution."""
  return np.random.normal(mean,scale,size)

def generate_B(mean,scale,size):
  """Creates a symmetric matrix B = A + A^T."""
  A=generate_A(mean,scale,size)
  B=A+A.T 
  return B

def eigen_values_vectors(B):
    """Computes EVD and sorts eigenvalues/vectors in descending order."""
    egnval,egnV=np.linalg.eigh(B) # eigh returns eigenvalues in ascending order
    egnval=np.flip(egnval)
    egnV=np.flip(egnV,axis=1)
    return egnval ,egnV 

def EVD(egnval,egnV):
  """Constructs the diagonal matrix D from eigenvalues."""
  return np.diag(egnval), egnV

# --- Part 2a: EVD Calculation ---
size=(100,100)
mean=0
sigma=math.sqrt(float(input("enter varience : ")))

# Generate a symmetric matrix B
B=generate_B(mean,sigma,size)
egnval , egnV = eigen_values_vectors(B)
digonalvector,eigenvectormatrix=EVD(egnval,egnV)

# Printing the components of the EVD
print("--- Results for Part 2a ---")
print("Diagonal eigenvalue matrix (D): \n", digonalvector)
print("\nEigenvector matrix (V): \n", eigenvectormatrix)

# Verify the decomposition B = V*D*V^T
Breconstructed=eigenvectormatrix @ digonalvector @ eigenvectormatrix.T
print("\nVerifying the decomposition. Is B == V*D*V^T? ->", np.allclose(Breconstructed,B))


# --- Part 2b: Low-Rank Approximation Error ---

def B_B_k_norm(B,k):
    """Calculates the error ||B - B_k||_F for a rank-k approximation."""
    egnval,egnV=eigen_values_vectors(B)
    B_k=np.zeros(B.shape)
    for i in range(k):
        v_i=np.expand_dims(egnV[:,i],axis=1)
        B_k+=egnval[i]*( v_i @ v_i.T )
    return float(np.linalg.norm(B-B_k,'fro'))

# Generate a new B matrix for this part
B=generate_B(mean,sigma,size)
k=int(input("enter value of K : "))
# Printing the Frobenius norm of the error
print("\n--- Results for Part 2b ---")
print(f"The error norm ||B - B_k||_F for k={k} is:", B_B_k_norm(B,k))


# --- Part 2c: Plotting Error Norm vs. k ---
print("\n--- Generating Plot for Part 2c ---")

# Generate a new B matrix for the plot
B=generate_B(mean,sigma,size)
For=[]
K=[]
for i in range(100):   
    K.append(i+1)
    For.append(B_B_k_norm(B,i))
    
# Plotting the error ||B - B_k||_F as a function of k
plt.plot(K,For)    
plt.xlabel("k (Number of Eigenvectors)")
plt.ylabel("Error Norm ||B - B_k||_F")
plt.title("Approximation Error vs. k")
plt.grid(True)
plt.show()


# --- Part 2d: Orthogonality of Eigenvectors ---

def VTV_I(egnV):
    """Checks for orthogonality by calculating ||V^T*V - I||_F."""
    I=np.identity(egnV.shape[1])
    return float(np.linalg.norm(egnV.T @ egnV - I,'fro'))

# Generate a new B matrix and find its eigenvectors
B=generate_B(mean,sigma,size)
egnval,egnV=eigen_values_vectors(B)
VTV_I_norm=VTV_I(egnV)

# Printing the result of the orthogonality check
print("\n--- Results for Part 2d ---")
print("Orthogonality check norm ||V^T*V - I||_F:", VTV_I_norm)
print("(A value close to zero indicates orthogonality)")


# --- Part 2e: Plotting Error Norm vs. Variance ---
print("\n--- Generating Plot for Part 2e ---")

def plot_data_for_B(k,size,mean):
    """Helper function to calculate error norms across a range of variances."""
    bbknorm=[]
    for i in range(101):
        sigma=math.sqrt(i*0.01)
        B=generate_B(mean,sigma,size)
        bbknorm.append(B_B_k_norm(B,k))
    return bbknorm

# Define the list of k values to plot
k=[5, 10, 20, 25, 30]
size=(100,100)
BigB=[]
varience=[]
for i in range(101):
    varience.append(i*0.01)
    
# Gather data for each k
for i in k:
    print(f"Calculating data for k={i}...")
    BigB.append(plot_data_for_B(i,size,mean))
    
# Plotting the error ||B - B_k||_F vs. variance for different k
for i in range(5):
    plt.plot(varience,BigB[i],label=f"k = {k[i]}")

plt.xlabel("Variance (σ^2)")
plt.ylabel("Error Norm ||B - B_k||_F")
plt.title("Approximation Error vs. Variance for different k")
plt.legend()
plt.grid(True)
plt.show()