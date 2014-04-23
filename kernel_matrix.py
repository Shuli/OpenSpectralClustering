# -*- coding: utf-8 -*-
"""
=============================================================================================
1.create_kernel_matrix
    Create a kernel matrix based on the vector specified
=============================================================================================
Operating conditions necessary {UTF-8/CrLf/Python2.7/numpy/matlot/Scipy}
@author: Hisashi Ikari
"""
import numpy as np

# ===========================================================================================
# Create a kernel matrix based on the vector specified
#　Please make sure that the kernel matrix is positive semi-definite matrix here
# -------------------------------------------------------------------------------------------
# Define the Laplacian kernel
# "lambda" is equivalent to the "define", accurately represents the lambda expression
# -------------------------------------------------------------------------------------------
# ***Definition of the kernel function***
#   k(x,z)=<φ(x), φ(z)>
#       x is the first column of the data file
#       z is the second column of the data file,
#       φ is the following definitions:
# -------------------------------------------------------------------------------------------
# *** Definition of the kernel matrix ***
#   When the kernel function k (·, ·) are given training set S and {x_l, x_1, ...}
#   K_ij = K(x_i, x_j)
#   K = the rbf kernel matrix ( = exp(-1/(2*sigma^2)*(coord*coord')^2) )
# ===========================================================================================
def create_kernel_matrix(A, sigma):
    N = len(A)
    
    # To calculate the dot product using the RBF kernel function
    K = np.dot(A, A.T) / sigma ** 2

    # To hold the diagonal elements    
    D = np.matrix(np.diag(K))

    # Set the exponent of the RBF
    T = np.dot(D.T, np.ones((1, N)) ).T
    K = K - T / 2
    K = K - T.T / 2
    K = np.exp(K)  

    return K
    
# ===========================================================================================
# Create a kernel matrix based on the vector specified
#　Please make sure that the kernel matrix is positive semi-definite matrix here
# ===========================================================================================
def create_kernel_matrix_by_dual(A, B, sigma):
    N = len(A)
    
    # To calculate the dot product using the RBF kernel function
    K = np.dot(A, B.T) / sigma ** 2

    # To hold the diagonal elements    
    D = np.matrix(np.diag(K))

    # Set the exponent of the RBF
    T = np.dot(D.T, np.ones((1, N)) ).T
    K = K - T / 2
    K = K - T.T / 2
    K = np.exp(K)  

    return K
    
