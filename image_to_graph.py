# -*- coding: utf-8 -*-
"""
===============================================================================
1.image_to_graph
    Maximum degree of the graph matrix is based on the amount of memory in your
    computer. Reduce size of image to fit the maximum degree of the graph.
    It takes the difference between the brightness of the adjacent pixel of 
    interest from the image, you can create a graph matrix.
===============================================================================
Operating conditions necessary {UTF-8/CrLf/Python2.7/Numpy/Matplot/Scipy}
@author: Hisashi Ikari
"""
import math
import numpy as np
import cv as cv

INDEX = 30
NEI8  = False

# =============================================================================
# image_to_graph
#    Maximum degree of the graph matrix is based on the amount of memory in
#    your computer. Reduce size of image to fit the maximum degree of the
#    graph. It takes the difference between the brightness of the adjacent 
#    pixel of interest from the image, you can create a graph matrix.
# =============================================================================
# *** Arguments ***
#   name: target image file name
# =============================================================================
def image_to_graph(name):
    # -------------------------------------------------------------------------
    # It converted to grayscale
    # -------------------------------------------------------------------------
    source = grayscale(name)
    #cv.Smooth(source, source, cv.CV_GAUSSIAN, INDEX / 2, INDEX / 2)

    # It defines the maximum dimension of the matrix, this is based on the 
    # amount of memory in your computer.
    sx = INDEX
    sy = INDEX
    st = sx * sy

    # -------------------------------------------------------------------------
    # Reduce size of image to fit the dimensions of the maximum of the graph.
    # -------------------------------------------------------------------------
    target = cv.CreateImage((sx, sy), cv.IPL_DEPTH_8U, 1);     
    cv.Resize(source, target)

    # -------------------------------------------------------------------------
    # it create a graph from the image that was reduced.
    # If you want to change type of pixel, then specify the dtype as follows
    #graph = np.zeros((st, st), dtype=np.int8)    
    # -------------------------------------------------------------------------
    graph = np.zeros((st, st))    

    # I'm sorry, the folowing is mistake now.
    # For instance, this is not a correct features.

    for y in range(1, sy - 1 - 1):
        for x in range(1, sx - 1 - 1):
            # -----------------------------------------------------------------
            # It will create a graph matrix based on the adjacent pixel
            # The definition of the direction of adjacent pixels, It defined
            # in the following
            # -----------------------------------------------------------------
            # 4 neighborhood:  8 neighborhood:
            # -----------------------------------------------------------------
            #   2              5 2 6 
            # 1 0 3            1 0 3  
            #   4              7 4 8
            # -----------------------------------------------------------------
            pixel_n0 = cv.Get2D(target, y,     x    )[0]
            
            # 4 neighborhood
            pixel_n1 = cv.Get2D(target, y,     x - 1)[0]
            pixel_n2 = cv.Get2D(target, y - 1, x    )[0]
            pixel_n3 = cv.Get2D(target, y,     x + 1)[0]
            pixel_n4 = cv.Get2D(target, y + 1, x    )[0]

            # 8 neighborhood
            if NEI8 == True:
                pixel_n5 = cv.Get2D(target, y - 1, x - 1)[0]
                pixel_n6 = cv.Get2D(target, y - 1, x + 1)[0]
                pixel_n7 = cv.Get2D(target, y + 1, x - 1)[0]
                pixel_n8 = cv.Get2D(target, y + 1, x + 1)[0]
            
            st = get_position(x, y, sx)
            graph[st, st] = pixel_n0
            
            # 4 neighborhood
            graph[st, get_position(x - 1, y    , sx)] = (pixel_n1 - pixel_n0)
            graph[st, get_position(x    , y - 1, sx)] = (pixel_n2 - pixel_n0)
            graph[st, get_position(x + 1, y    , sx)] = (pixel_n3 - pixel_n0)
            graph[st, get_position(x    , y + 1, sx)] = (pixel_n4 - pixel_n0)

            # 8 neighborhood
            if NEI8 == True:
                graph[st, get_position(x - 1, y - 1, sx)] = \
                    (pixel_n5 - pixel_n0)
                graph[st, get_position(x + 1, y - 1, sx)] = \
                    (pixel_n6 - pixel_n0)
                graph[st, get_position(x - 1, y + 1, sx)] = \
                    (pixel_n7 - pixel_n0)
                graph[st, get_position(x + 1, y + 1, sx)] = \
                    (pixel_n8 - pixel_n0)

    # And put it back in the file temporarily graph obtained
    np.savetxt("result_graph.dat", graph)
    
    return graph

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
# This is test for creation graph from image
# =============================================================================
# -----------------------------------------------------------------------------
# Initial processing
# -----------------------------------------------------------------------------
#image_to_graph("C:\\Python27\\Lib\\site-packages\\xy\\101_ObjectCategories\\dalmatian\\image_0003.jpg")
#image_to_graph("C:\\Python27\\Lib\\site-packages\\xy\\101_ObjectCategories\\garfield\\image_0020.jpg")
#image_to_graph("C:\\Python27\\Lib\\site-packages\\xy\\books.png")
#image_to_graph("C:\\Python27\\Lib\\site-packages\\xy\\base.PNG")
#image_to_graph("C:\\Python27\\Lib\\site-packages\\xy\\result_gradiation.png")
#image_to_graph("C:\\Python27\\Lib\\site-packages\\xy\\circle.png")
    
