# 🐍 CSL2010 Lab Scripts

A collection of Python scripts for solving a series of linear algebra and data analysis problems. This repository includes solutions for calculating the Frobenius norm, performing Eigenvalue Decomposition (EVD), and conducting a statistical analysis of the Iris dataset.

---

### 📝 `q1.py` - Frobenius Norm Calculator

This script calculates the Frobenius norm of a randomly generated matrix **without** using built-in library functions for the norm itself.

#### Structure & Usage
* **`generate_A()`**: Creates a matrix with entries from a Gaussian (normal) distribution.
* **`froba_normal()`**: Manually calculates the Frobenius norm by iterating through every element, squaring it, summing the results, and taking the final square root.

#### 🚀 How to Run
1.  Execute the script from your terminal:
    ```bash
    python q1.py
    ```
2.  When prompted, enter integer values for the matrix dimensions **`m`** (must be > 6) and **`n`** (must be > 8).
3.  The script will output the final calculated norm.

---

### 📊 `q2.py` - Eigenvalue Decomposition (EVD) Analysis

This script provides a comprehensive analysis of EVD, low-rank matrix approximation, and eigenvector properties using `numpy` and `matplotlib`.

#### Structure & Usage
* **`generate_B()`**: Creates a random **symmetric** matrix `B` for analysis.
* **`eigen_values_vectors()`**: Computes the eigenvalues and eigenvectors of `B`, sorting them in descending order.
* **`B_B_k_norm()`**: Calculates the approximation error `||B - B_k||_F` for a given rank `k`.

#### 🚀 How to Run
1.  Run the script from your terminal:
    ```bash
    python q2.py
    ```
2.  The script will guide you through the different parts of the question, prompting for input as needed:
    * Enter a **variance** (e.g., `1.0`) for the initial EVD calculation.
    * Enter an integer **`k`** for the low-rank approximation error calculation.
3.  The script will print analytical results to the console and generate two plots:
    * **Plot 1:** Approximation Error vs. `k`.
    * **Plot 2:** Approximation Error vs. Matrix Element Variance.

---

### 🌸 `q3.py` - Iris Dataset Statistical Analysis

This script uses `pandas` and `numpy` to perform a statistical breakdown of the classic Iris dataset.

> **Note:** This script requires the **`iris.txt`** file to be in the same directory to run successfully.

#### Structure & Usage
This script runs procedurally from top to bottom, performing and printing the results for each analysis step:
1.  Loads the **`iris.txt`** dataset.
2.  Calculates and prints the **Covariance Matrix**.
3.  Calculates and prints the **Correlation Matrix** and identifies the feature pairs with the highest, lowest, and most negative correlation.
4.  Calculates and prints the **Sample Mean and Variance** for each feature.
5.  Calculates and prints the **Eigenvalues** of the covariance matrix.

#### 🚀 How to Run
1.  Ensure **`iris.txt`** is in the same folder as the script.
2.  Execute the script:
    ```bash
    python q3.py
    ```
3.  All results will be printed directly to the console. No user input is required.