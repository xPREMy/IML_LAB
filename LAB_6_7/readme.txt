README: K-Means and Spectral Clustering Lab
This document provides instructions on how to run the Python scripts for the CSL2010 Lab 6 & 7 assignments.

1. Prerequisites
Before running the scripts, ensure you have Python installed along with the following libraries:

pandas

numpy

matplotlib

You can install these packages using pip:

pip install pandas numpy matplotlib

2. File Structure
For the scripts to run correctly, your files must be organized in the same directory as follows:

/your_project_folder
|-- Q1.py
|-- Q2.py
|-- dataset.txt

The dataset.txt file is crucial as it contains the data used for clustering.

3. Running the Code
You can run the scripts from your terminal or command prompt.

K-Means Clustering (Q1.py)
This script implements the K-Means clustering algorithm directly on the dataset.

To run it, navigate to the project folder in your terminal and execute:

python Q1.py

Expected Output:

The script will print the progress of the clustering epochs to the console.

A plot will be displayed showing the two clusters found by the K-Means algorithm, colored red and green.

The final accuracy of the clustering will be printed to the console.

Spectral Clustering (Q2.py)
This script implements the Spectral Clustering algorithm, which involves creating a similarity graph and then applying K-Means on the spectral embeddings.

To run it, execute the following command in your terminal:

python Q2.py

Expected Output:

The script will first display a plot of the spectral embeddings, showing how the data is transformed to become linearly separable.

After closing the first plot, a second plot will appear showing the final clusters on the original data, colored red and green.

The final accuracy (which should be 100%) will be printed to the console.