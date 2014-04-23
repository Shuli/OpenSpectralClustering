# -*- coding: utf-8 -*-
"""
===============================================================================
1.spectral_clustering_for_image
    Binarization of the image is performed by spectral clustering.
===============================================================================
Operating conditions necessary {UTF-8/CrLf/Python2.7/Numpy/Matplot/Scipy}
@author: Hisashi Ikari
"""
import numpy as np
import image_to_graph as ig
import graph_to_image as gi
import spectral_clustering_nosparse as sc
import kernel_matrix as km

# =============================================================================
# spectral_clustering_for_image
#   Binarization of the image is performed by spectral clustering.
# =============================================================================
# -----------------------------------------------------------------------------
# It convert image to the graph
# -----------------------------------------------------------------------------
ig.image_to_graph("C:\\Python27\\Lib\\site-packages\\xy\\base.PNG")

C = 100

# -----------------------------------------------------------------------------
# If you want to classify random data from a normal distribution
# -----------------------------------------------------------------------------
A = np.loadtxt("result_graph.dat")
K = km.create_kernel_matrix(A, 5.0)
R,mi = sc.spectral_clustering_nosparse(K,C,True,True)
C,N  = sc.create_classification(K, R, mi, "result_spectral_clustering.txt")

sc.draw_classification(R, C, N)

# -----------------------------------------------------------------------------
# It convert graph to the image
# -----------------------------------------------------------------------------
gi.graph_to_image("C:\\Python27\\Lib\\site-packages\\xy\\base.PNG")    
