# -*- coding: utf-8 -*-
"""
===============================================================================
1.graph_to_image
    Draw the image classification which is obtained by using the spectral
    clustering.
===============================================================================
Operating conditions necessary {UTF-8/CrLf/Python2.7/Numpy/Matplot/Scipy}
@author: Hisashi Ikari
"""
import math
import numpy as np
import cv

INDEX = 30

# =============================================================================
# graph_to_image
#    Draw the image classification which is obtained by using the spectral
#    clustering.
# =============================================================================
# *** Arguments ***
#   name: target image(source image) file name
# =============================================================================
def graph_to_image(name):
    
    A = np.loadtxt("result_graph.dat")
    B = np.loadtxt("result_spectral_clustering.txt")

    source = cv.LoadImage(name, cv.CV_LOAD_IMAGE_COLOR)
    tx = INDEX
    ty = INDEX

    target = cv.CreateImage((tx, ty), cv.IPL_DEPTH_8U, 3);     
    cv.Resize(source, target)

    # -------------------------------------------------------------------------
    # I draw to the image classification obtained.
    # -------------------------------------------------------------------------
    N = INDEX
    for i in range(0, N - 1):
        for j in range(0, N - 1):    
            t = get_position(j, i, INDEX)
            r, g, b = ((0, 0, 0) if float(B[t]) == 0.0 \
                                 else color(int(B[t]), INDEX * INDEX) )
            cv.Set2D(target, i, j, cv.Scalar(r, b, g))

    
    result = cv.CreateImage((source.width, source.height), cv.IPL_DEPTH_8U, 3);     
    cv.Resize(target, result)
    cv.SaveImage("result_source.png", source)
    cv.SaveImage("result_result.png", result)

    # -------------------------------------------------------------------------
    # I will draw the image pixels of each classification for debugging.
    # -------------------------------------------------------------------------
    C = np.unique(B)
    M = len(C)    
    for h in C:
        temp = cv.CreateImage((tx, ty), cv.IPL_DEPTH_8U, 3);
        cv.SetZero(temp)
        for i in range(0, N - 1):
            for j in range(0, N - 1):
                t = get_position(j, i, INDEX)
                if (float(B[t]) == h):
                    cv.Set2D(temp, i, j, cv.Scalar(255, 255, 255))       
                    
        cv.SaveImage("result_cluster_%d.png" % (h), temp)
        print("result_cluster_%d.png" % (h))

# =============================================================================
# get_position
#    I converted to a position in the array of one-dimensional position of the
#    image of(from) the two-dimensional.
# =============================================================================
# *** Arguments ***
#   x     : x-position on two-demensinal image
#   y     : y-position on two-demensinal image
#   size_x: width of x-position on two-demensinal image
# =============================================================================
def get_position(x, y, size_x):
    return y * size_x + x

# =============================================================================
# grayscale
#    I will convert the image of the gray scale image of RGB.
# =============================================================================
# *** Arguments ***
#   name: target image file name
# =============================================================================
def grayscale(name):
    img = cv.LoadImage(name, cv.CV_LOAD_IMAGE_COLOR)
    grayscaled_img = cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_8U, 1)
    cv.CvtColor(img, grayscaled_img, cv.CV_BGR2GRAY)
    return grayscaled_img

# =============================================================================
# color
#    I will create a color based on the classification.
# =============================================================================
# *** Arguments ***
#   target : target classification number
#   maximum: maimum classification number
# =============================================================================
def color(target, maximum):
    h = [0, 0, 0]
    sg = 256 * 6 / (maximum) * (target)
    if sg >= 0 and sg <=(256 * 1 - 1):
        h[0] = 255
        h[1] = sg
        h[2] = 0
    elif sg > (256 * 1 - 1) and sg <= (256 * 2 - 1):
        h[0] = 256 * 2 - 1 - sg
        h[1] = 255
        h[2] = 0
    elif sg > (256 * 2 - 1) and sg <= (256 * 3 - 1):
        h[0] = 0
        h[1] = 255
        h[2] = sg - 256 * 2
    elif sg > (256 * 3 - 1) and sg <= (256 * 4 - 1):
        h[0] = 0
        h[1] = 256 * 4 - 1 - sg
        h[2] = 255
    elif sg > (256 * 4 - 1) and sg <= (256 * 5 - 1):
        h[0] = sg - 256 * 4
        h[1] = 0
        h[2] = 255
    elif sg > (256 * 5 - 1) and sg <= (256 * 6 - 1):
        h[0] = 255
        h[1] = 0
        h[2] = 256 * 6 - 1 - sg

    if h[2] == 0 and h[1] == 0 and h[0] == 0:
        h[2] = 255
        h[1] = 255
        h[0] = 255

    return h[2], h[1], h[0]
    
# =============================================================================
# This is test for creation image from graph
# =============================================================================
# -----------------------------------------------------------------------------
# Initial processing
# -----------------------------------------------------------------------------
#graph_to_image("C:\\Python27\\Lib\\site-packages\\xy\\101_ObjectCategories\\dalmatian\\image_0003.jpg")
#graph_to_image("C:\\Python27\\Lib\\site-packages\\xy\\101_ObjectCategories\\garfield\\image_0020.jpg")    
#graph_to_image("C:\\Python27\\Lib\\site-packages\\xy\\books.png")    
#graph_to_image("C:\\Python27\\Lib\\site-packages\\xy\\base.PNG")    
#graph_to_image("C:\\Python27\\Lib\\site-packages\\xy\\result_gradiation.png")    
#graph_to_image("C:\\Python27\\Lib\\site-packages\\xy\\circle.png")



