# -*- coding: utf-8 -*-
"""
=============================================================================================
1.spectral_clustering
    And the adjacency matrix of the graph,
    from the dot product of the information given in the direction of that from vertex to
    vertex, perform the classification by K-Means.
=============================================================================================
Operating conditions necessary {UTF-8/CrLf/Python2.7/numpy/matlot/Scipy}
@author: Hisashi Ikari
"""
# Standard Library
import numpy as np
import scipy as sp
from scipy import *
from scipy.sparse import spdiags, issparse
from scipy.linalg import *
from matplotlib.pyplot import *

# For Plotgraph
import matplotlib.font_manager as fm
import scipy.interpolate as intp

# Custom Library
import kernel_matrix as km # Create a kernel matrix using a Gaussian kernel function
import dualkmeans as dkm   # Based on the duality matrices, according to the k-means then the classification

# ===========================================================================================
# spectral_clustgering
#    And the adjacency matrix of the graph,
#    from the dot product of the information given in the direction of that from vertex to
#    vertex, perform the classification by K-Means.
# ===========================================================================================
# *** Arguments ***
#   K  : kernel matrix
#   c  : number of clusters you want to split  
#   kmf: Whether or not to use the k-means classification
#   ptf: Whether or not you want to graph the results
#   ds : Forced to use the specified dimension{True, False}
# ===========================================================================================
def spectral_clustering(K,c,kmf,ptf,ds=False):

    # Kernel matrix is ​​always positive semi-definite in the square    
    N = len(K) 
    R = 100

    # Reconnect to the matrix diagonal matrix    
    # However, if you match the number of dimensions of the sparse matrix
    D = spdiags([sum(K[i,:],0,N,N) for i in xrange(N)],0,N,N) if issparse(K) else \
           diag([sum(K[i,:]) for i in xrange(N)])
    
    if issparse(K):    
        print("matrix is sparse")    
    
    # ---------------------------------------------------------------------------------------
    # The normalization by "summing"    
    # ---------------------------------------------------------------------------------------
    # If you want to perform normalization by summing, 
    # we calculate the following processing RH
    #   RH = speye(N) if issparse(K) else eye(N) # summing
    #   RH = D                                   # subtraction
    # ---------------------------------------------------------------------------------------
    LH = D - K
    RH = np.eye(N) if issparse(K) else np.eye(N)

    # ---------------------------------------------------------------------------------------
    # The invitation by the eigenvalue decomposition, the eigenvalues ​​and eigenvectors
    # ---------------------------------------------------------------------------------------
    ev,ed = sp.linalg.eig(LH, RH) # returns  eigen-values, eigen-vectors
    #ev,ed = np.linalg.eig(K) # returns  eigen-values, eigen-vectors
    print "ev-1\n", (ev)
    print "ed-1\n", (ed)

    #savetxt("result_source_ev.txt", ev, fmt="%f")
    #savetxt("result_source_ed.txt", ed, fmt="%f")

    # Locate the lowest eigenvalue, and then compressed to a dimension that the eigenvectors
    md = min(abs(ev))
    mi = aindex(ev, md, len(ev))
    if ds == True:
        mi = c if c < mi else mi
    ed = ed[:,0:mi]

    #savetxt("result_result_ed.txt", ed, fmt="%f")

    # Perform the normalization denominator and the norm
    for i in range(mi):
        ed[:,i] = ed[:,i] / norm(ed[:,i])

    # Calculation is performed to obtain the distance of the cluster
    dv = sqrt(diag(dot(ed,ed.T))) + 1.0e-8;
    vo = dot(diag(1. / dv), ed)

    #savetxt("result_result_vo.txt", vo.T, fmt="%f")
    #savetxt("vo.dat", vo, fmt="%f")
    #savetxt("rand.dat", [ceil(np.random.random() * N) for i in range(R)], fmt="%d")
    
    # ---------------------------------------------------------------------------------------
    # perform the classification by k-means
    # ---------------------------------------------------------------------------------------
    # In this case, asking when processed by the function k-means "in Matlab", 
    # the initial conditions. Because there is no such function, 
    # the following processing must be performed does not exist in Python.
    # ---------------------------------------------------------------------------------------
    if False: #kmf == True:
        cent = np.zeros((R,C,C))
        for i in range(R):
            tent = floor(np.random.random() * N)
            nent = matrix(vo[tent,:])
        
            for j in range(1,C):
                dotp = dot(vo, nent.T)
                tmax = [amax(abs(dotp[r])) for r in range(N)]
                tmin = amin(tmax)                
        
                for p in range(N):
                    if tmax[p] == tmin:    
                        nnum = p
                        break
        
                nent.resize(j + 1, C)
                nent[j] = vo[nnum,:]
        
            cent[i] = nent
    
    return vo,mi

# ===========================================================================================
# aindex
#   In Matlab, there has the min(A, [], 2), But Python does not have.
#   Therefore, it returns the number of the array of elements with the value of the minimum
# ===========================================================================================
# *** Arguments ***
#   A:matrix to compare 1
#   B:target number
#   C:number of array
# ===========================================================================================
def aindex(A, b, c):
    for i in xrange(c):
        if A[i] == b:
            return i
    return -1

# ===========================================================================================
# Reads the data file
# Data file is a newline character to separate the code that depends on the OS space and,
# given x, y, label {0,1}
# ===========================================================================================
# -------------------------------------------------------------------------------------------
# Initial processing
# -------------------------------------------------------------------------------------------
C = 10

# If you want to classify random data from a normal distribution
K = np.loadtxt("result_graph.dat")
#K = km.create_kernel_matrix(A,1.0)
#A = np.loadtxt("training.dat")

#A = np.loadtxt("result_graph.dat")
#K = km.create_kernel_matrix(A,1.0)

print("graph loaded")

# If you want to classify the vertices of a graph using the graph kernel
#K = np.loadtxt("graph_A.txt")
R,mi = spectral_clustering(K,C,True,True)

print("graph clustered")

R = R.real
print "R\n", R


# The distance from the cluster obtained,
# it will calculated the cluster value belongs.
N = len(K)
C = mi
RI = zeros((N))
for i in range(N):
    #     
    mt = 0.0
    asc = np.sort(abs(R[i,:]))
    for j in asc:
        if (j != 0.0):
            mt = j 
            break
    
    print("sort:")
    print(np.sort(abs(R[i,:])))
    print("result:")
    print(mt)    
    
    #mt = min(abs(R[i,:]))
    #print "i,mt\n", i, mt
    ri = aindex(abs(R[i]),mt,C) 
    RI[i] = ri
    print "index-%d=(result:%i,distance:%f)" % (i, ri, mt)

savetxt("result_spectral_clustering.txt", RI.real, fmt="%d")

print("graph saved")

# -------------------------------------------------------------------------------------------
# Draw the graph
# -------------------------------------------------------------------------------------------
figure(num=None, figsize=(14, 6), facecolor='w', edgecolor='k')

for i in range(C):
    plot(R[:,i],"o-",label=i+1)

grid(True)

legend(["cluster-%d" % (i + 1) for i in range(C)], \
        prop=fm.FontProperties(size=10.5), \
        bbox_to_anchor=(1.01, 1.0), loc=2, borderaxespad=0.0)

title("Classification as the spectrum of spectral clustering")
xlabel("Node number(%d nodes)" % N)
ylabel("Distance from the center of each node(%d clusters)" % C)

show()
close()
