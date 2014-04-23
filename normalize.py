# -*- coding: utf-8 -*-
"""
=============================================================================================
1.normalize_matrix
    To center the kernel matrix
=============================================================================================
Operating conditions necessary {UTF-8/CrLf/Python2.7/numpy/matlot/Scipy}
@author: Hisashi Ikari
"""
import numpy as np

# ===========================================================================================
# To center the kernel matrix
# Kernel matrix can be input is a square matrix
# This process, "which is based on the distance of the center of mass",
# rather than the normalization is normalizationdata type not understood
# -------------------------------------------------------------------------------------------
# *** Definition of the centering ***
#   k(x,z) = k(x,z) - (1/l)Σ_l k(x,x_i) - (1/l)Σ_l k(z,x_i) + (1/l^2)Σ_l k(x_i, z_i)
# ===========================================================================================
def normalize_matrix(K):
    # Holds the size of the array    
    N = len(K) 

    # Create a vector with the average of the "columns" of the matrix kernel
    D = np.zeros((N)) 
    for i in range(N):
        D[i] = np.mean(K[i,:])

    # Scalar value of the average of all the components of K
    E = np.sum(D) / N
    J = np.ones((N, 1)) * D

    Kc = K - J - J.T + E * np.ones((N, N))
    
    return Kc
