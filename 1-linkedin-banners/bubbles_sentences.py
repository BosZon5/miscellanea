#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities to print strings with a bubbles style, where each trait is made up by
a cloud of bubbles. 

@author: Andrea Boselli
"""

#%% Import libraries, and define variables and functions

import numpy             as np
import matplotlib.pyplot as plt
import sympy             as sp

from sympy.abc import t



letterShapes = {
                "A": [sp.Curve((t/2,       t  ), (t, 0, 1)),
                      sp.Curve((1/2 + t/2, 1-t), (t, 0, 1)),
                      sp.Curve((1/4 + t/2, 1/2), (t, 0, 1))],
            
                "B": [sp.Curve((0,                             t                            ), (t, 0, 1)),
                      sp.Curve((t/2,                           1                            ), (t, 0, 1)),
                      sp.Curve((t/2,                           1/2                          ), (t, 0, 1)),
                      sp.Curve((1/2 + sp.cos((3/2+t)*sp.pi)/2, 3/4 + sp.sin((3/2+t)*sp.pi)/4), (t, 0, 1)),
                      sp.Curve((t/2,                           0                            ), (t, 0, 1)),
                      sp.Curve((1/2 + sp.cos((3/2+t)*sp.pi)/2, 1/4 + sp.sin((3/2+t)*sp.pi)/4), (t, 0, 1))],
            
                "C": [sp.Curve((sp.cos((1/2+t)*sp.pi)+1, 1/2 + sp.sin((1/2+t)*sp.pi)/2), (t, 0, 1))],
            
                "D": [sp.Curve((0,                     1-t                            ), (t, 0, 1)),
                      sp.Curve((sp.cos((3/2+t)*sp.pi), 1/2 + sp.sin((3/2+t)*sp.pi)/2  ), (t, 0, 1))],
            
                "E": [sp.Curve((t,   0  ), (t, 0, 1)),
                      sp.Curve((t,   1  ), (t, 0, 1)),
                      sp.Curve((t/2, 1/2), (t, 0, 1)),
                      sp.Curve((0,   1-t), (t, 0, 1))],
            
                "F": [sp.Curve((t,   1  ), (t, 0, 1)),
                      sp.Curve((t/2, 1/2), (t, 0, 1)),
                      sp.Curve((0,   t  ), (t, 0, 1))],
            
                "G": [sp.Curve((sp.cos((1/2+t)*sp.pi)+1, 1/2 + sp.sin((1/2+t)*sp.pi)/2), (t, 0, 1)),
                      sp.Curve((1,                       t/2                          ), (t, 0, 1)),
                      sp.Curve((1-t/2,                   1/2                          ), (t, 0, 1))],
            
                "H": [sp.Curve((0, t  ), (t, 0, 1)),
                      sp.Curve((1, t  ), (t, 0, 1)),
                      sp.Curve((t, 1/2), (t, 0, 1))],
                
                "I": [sp.Curve((t,   0), (t, 0, 1)),
                      sp.Curve((t,   1), (t, 0, 1)),
                      sp.Curve((1/2, t), (t, 0, 1))],
            
                "J": [sp.Curve((t,                           1),       (t, 0, 1)),
                      sp.Curve((1/2,                         1/4+t*3/4), (t, 0, 1)),
                      sp.Curve((1/4 + sp.cos(sp.pi*(1+t))/4, 1/4+sp.sin(sp.pi*(1+t))/4), (t, 0, 1))],
            
                "K": [sp.Curve((0, t      ), (t, 0, 1)),
                      sp.Curve((t, 1/2+t/2), (t, 0, 1)),
                      sp.Curve((t, 1/2-t/2), (t, 0, 1))],
            
                "L": [sp.Curve((0, 1-t), (t, 0, 1)),
                      sp.Curve((t,   0), (t, 0, 1))],
            
                "M": [sp.Curve((0,       t      ), (t, 0, 1)),
                      sp.Curve((1,       t      ), (t, 0, 1)),
                      sp.Curve((t/2,     1-t/2  ), (t, 0, 1)),
                      sp.Curve((1/2+t/2, 1/2+t/2), (t, 0, 1))],
            
                "N": [sp.Curve((0, t  ), (t, 0, 1)),
                      sp.Curve((t, 1-t), (t, 0, 1)),
                      sp.Curve((1, t  ), (t, 0, 1))],
            
                "O": [sp.Curve((1/2 + sp.cos(2*sp.pi*t)/2, 1/2 + sp.sin(2*sp.pi*t)/2), (t, 0, 1))],
            
                "P": [sp.Curve((0,                             t                            ), (t, 0, 1)),
                      sp.Curve((t/2,                           1                            ), (t, 0, 1)),
                      sp.Curve((t/2,                           1/2                          ), (t, 0, 1)),
                      sp.Curve((1/2 + sp.cos((3/2+t)*sp.pi)/2, 3/4 + sp.sin((3/2+t)*sp.pi)/4), (t, 0, 1))],
            
                "Q": [sp.Curve((1/2 + sp.cos(2*sp.pi*t)/2,         1/2 + sp.sin(2*sp.pi*t)/2), (t, 0, 1)),
                      sp.Curve((1/sp.sqrt(2) + t*(1-1/sp.sqrt(2)), (1-t)*(1-1/sp.sqrt(2))   ), (t, 0, 1))],
            
                "R": [sp.Curve((0,                             t                            ), (t, 0, 1)),
                      sp.Curve((t/2,                           1                            ), (t, 0, 1)),
                      sp.Curve((t/2,                           1/2                          ), (t, 0, 1)),
                      sp.Curve((1/2 + sp.cos((3/2+t)*sp.pi)/2, 3/4 + sp.sin((3/2+t)*sp.pi)/4), (t, 0, 1)),
                      sp.Curve((1/2 + t/2,                     1/2 - t/2                    ), (t, 0, 1))],
                
                "S": [sp.Curve((1/2 + sp.cos((      3/2*t)*sp.pi)/2, 3/4 + sp.sin((      3/2*t)*sp.pi)/4), (t, 0, 1)),
                      sp.Curve((1/2 + sp.cos((1/2 - 3/2*t)*sp.pi)/2, 1/4 + sp.sin((1/2 - 3/2*t)*sp.pi)/4), (t, 0, 1))],
                
                "T": [sp.Curve((t,   1), (t, 0, 1)),
                      sp.Curve((1/2, t), (t, 0, 1))],
                
                "U": [sp.Curve((1/2 + sp.cos(sp.pi*(1+t))/2, 1/2 + sp.sin(sp.pi*(1+t))/2), (t, 0, 1)),
                      sp.Curve((0,                           1-t/2                      ), (t, 0, 1)),
                      sp.Curve((1,                           1/2+t/2                    ), (t, 0, 1))],
                
                "V": [sp.Curve((t/2,     1-t), (t, 0, 1)),
                      sp.Curve((1/2+t/2, t  ), (t, 0, 1))],
                
                "W": [sp.Curve((t/4,     1-t    ), (t, 0, 1)),
                      sp.Curve((1/4+t/4, t/2    ), (t, 0, 1)),
                      sp.Curve((1/2+t/4, 1/2-t/2), (t, 0, 1)),
                      sp.Curve((3/4+t/4, t      ), (t, 0, 1))],
                
                "X": [sp.Curve((t/2,     1-t/2  ), (t, 0, 1)),
                      sp.Curve((1/2+t/2, 1/2+t/2), (t, 0, 1)),
                      sp.Curve((t/2,     t/2    ), (t, 0, 1)),
                      sp.Curve((1-t/2,   t/2    ), (t, 0, 1))],
                
                "Y": [sp.Curve((t/2,     1-t/2  ), (t, 0, 1)),
                      sp.Curve((1/2+t/2, 1/2+t/2), (t, 0, 1)),
                      sp.Curve((1/2,     t/2    ), (t, 0, 1))],
                
                "Z": [sp.Curve((t, 0), (t, 0, 1)),
                      sp.Curve((t, 1), (t, 0, 1)),
                      sp.Curve((t, t), (t, 0, 1))]
                }
# letterShapes is a dictionary with the uppercase letters as key, and a list as value;
# each list item is a SymPy parametric curve describing a specifit trait of the letter


def letterPoints(letter, 
                 points_per_unit = 50, 
                 letter_width    =  1, 
                 letter_height   =  1):
    
    """
    Retrieve points on the traits of a letter.
    
    Parameters
    ----------
    letter : string
        Character of the letter to be returned. Only uppercase letters are allowed
    
    points_per_unit : numeric
        Positive number specifying the number of points per unit length of trait
    
    letter_width : numeric
        Positive number specifying the width of the character
        
    letter_height : numeric
        Positive number specifying the height of the character
    
    Returns
    -------
    out1 : numpy.ndarray
        x-coordinates of each point
        
    out2 : numpy.ndarray
        y-coordinates of each point
        
    out3 : numpy.ndarray
        Number of points for each trait of the letter
    """
        
    # The space yields no points
    
    if(letter == ' '):
        return np.array([]), np.array([]), np.array([],dtype=np.intc)
    
    # Get the parametric curves of the letters and initialize the coordinates lists
    
    letterShape = letterShapes[letter]
    
    xs = []
    ys = []
    ns = []
    
    # Iterate on each parametric curve
    
    for curve in letterShape:
        
        # Scale the curve and compute its length
        
        curve     = curve.scale(x = letter_width, 
                                y = letter_height)
        curve_len = float(curve.length)
        
        # Number of points and their parametrizations in [0,1]

        points_num = int(np.round(points_per_unit * curve_len))
        points_t   = np.linspace(start=0, stop=1, num=points_num, endpoint=True)
        
        # Extract the NumPy equivalent functions and compute the points
        
        x_t = sp.lambdify(t, curve.functions[0], 'numpy') # x(t)
        y_t = sp.lambdify(t, curve.functions[1], 'numpy') # y(t)
        
        xs.append(x_t(points_t) + np.zeros(points_num)) # x coordinates of the curve points
        ys.append(y_t(points_t) + np.zeros(points_num)) # y coordinates of the curve points
        ns.append(points_num)                           # number of curve points
        
    # Return the concatenated points and their number for all the curves
    
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), np.array(ns,dtype=np.intc)
        


def sentencePoints(sentence, 
                   points_per_unit = 50, 
                   letter_width    =  1, 
                   letter_height   =  2, 
                   space_width     =  0.4):
    """
    For each letter in the input sentence, retrieve the points on its traits.
    
    Parameters
    ----------
    sentence : string
        Sentence to be processed. Only uppercase letters and spaces are allowed
        
    points_per_unit : numeric
        Positive number specifying the number of points per unit length of trait
    
    letter_width : numeric
        Positive number specifying the width of the characters
        
    letter_height : numeric
        Positive number specifying the height of the characters
        
    space_width : numeric
        Positive number specifying the spacing between characters
    
    Returns
    -------
    out1 : numpy.ndarray
        x-coordinates of each point
        
    out2 : numpy.ndarray
        y-coordinates of each point
        
    out3 : numpy.ndarray
        Number of points for each (non-space) letter. The spaces are not 
        considered as letter and, thus, yield no number
    """
    
    # Initialize the current x coordinate and the coordinates lists

    x_curr = 0 # sliding x coordinate

    xs = [] # x coordinates
    ys = [] # y coordinates
    ns = [] # n points
    
    letters_list = list(sentence) # list of letters
    
    # Iterate on each letter
    
    for letter in letters_list:
        
        x,y,n = letterPoints(letter,points_per_unit, letter_width, letter_height) # retrive the letter points

        xs.append(x + x_curr)    # save the points x coordinates
        ys.append(y)             # save the points y coordinates
        
        if(n.size > 0):
            ns.append(np.sum(n)) # save the number of points per letter
        
        x_curr += letter_width + space_width
        
    # Concatenate and return the x and y coordinates of the sentence; return also the number of points for each printed letter
    
    if(not xs):
        return np.array([]), np.array([]), np.array([])
    
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), np.array(ns)



def printSentence(axis, 
                  sentence, 
                  points_per_unit   = 50, 
                  letter_width      =  1, 
                  letter_height     =  2, 
                  space_width       =  0.4, 
                  x_start           =  0, 
                  y_start           =  0, 
                  points_dispersion =  0, 
                  points_size       =  1, 
                  points_cmap       = 'hsv'):
    
    """
    Prints  an  input  sentence  with  matplotlib  using the dots style. Multiple
    style settings can be chosen with this function. 
    
    Parameters
    ----------
    axis : matplotlib.axes
        Axes objects representing the image on which the sentence must be printed
        
    sentence : string
        Sentence to be printed. Only uppercase letters and spaces are allowed
        
    points_per_unit : numeric
        Positive number specifying the number of points per unit length of trait
    
    letter_width : numeric
        Positive number specifying the width of the characters
        
    letter_height : numeric
        Positive number specifying the height of the characters
        
    space_width : numeric
        Positive number specifying the spacing between characters
        
    x_start : numeric
        x-coordinate of the lower-left corner of the printed sentence
        
    y_start : numeric
        y-coordinate of the lower-left corner of the printed sentence
    
    points_dispersion : numeric
        Positive number specifying the dispersion of the points from the letters
        traits
    
    points_size : numeric
        Positive number that influences the size of the points; notice that 
        each point has a random size anyway
        
    points_cmap : string
        String of the matplotlib colormap used to sample the points colors
    
    Returns
    -------
    
    out : numpy.ndarray
        NumPy array with, in order, left and right limits and lower and upper
        limits of the plot
    """
    
    # Retrieve the coordinates and the number of points per letter
    
    x_coords, y_coords, n_points = sentencePoints(sentence, points_per_unit, letter_width, letter_height, space_width)
    
    # Print the points with the chosen settings
    
    axis.scatter(x          = x_coords + x_start + np.random.randn(sum(n_points))*points_dispersion,
                  y          = y_coords + y_start + np.random.randn(sum(n_points))*points_dispersion,
                  c          = np.random.randint(0, sum(n_points), sum(n_points)),
                  edgecolors = 'none',
                  s          = np.abs(np.random.randn(sum(n_points)))*points_size,
                  cmap       = points_cmap)
    
    return(np.array([x_start, 
                     x_start + len(sentence) * (letter_width+space_width) - space_width,
                     y_start,
                     y_start + letter_height]))



#%% Example execution

if __name__ == '__main__':
    
    # Seed settings
    
    fix_seed = True # whether to fix the NumPy seed or not
    seed     = 1    # NumPy seed, in case it is fixed
    
    # Single letter plot settings
    
    single_letter       = 'X'       # letter to be plotted
    points_per_unit_l   =  100      # bubbles per unit length of trait
    
    letter_width_l      = 1         # letter width
    letter_height_l     = 2         # letter height
    space_width_l       = 0.4       # spaces width (non important here)
    points_dispersion_l = 1/20      # bubbles dispersion around the letter traits
    points_size_l       = 9         # parameter influencing the bubbles sizes
    points_cmap_l       = 'gnuplot' # colormap used to color the bubbles
    
    # Sentence plot settings
    
    sentence            = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" # sentence to be plotted
    points_per_unit_s   = 45                           # bubbles per unit length of trait
    
    letter_width_s      = 1         # letters width
    letter_height_s     = 2         # letters height
    space_width_s       = 0.4       # spaces width
    points_dispersion_s = 1/20      # bubbles dispersion around the letters traits
    points_size_s       = 1/2       # parameter influencing the bubbles sizes
    points_cmap_s       = 'gnuplot' # colormap used to color the bubbles
    
    # Common settings
    
    x_start = 0 # x-coordinate of the lower-left corner of the sentence
    y_start = 0 # y-coordinate of the lower-left corner of the sentence
    
    # Image construction for one single letter
    
    if(fix_seed):            # if required, fix the NumPy seed
        np.random.seed(seed)
    
    fig, ax = plt.subplots(dpi=1000)
    ax.set_aspect('equal', 'box')
    ax.set_axisbelow(True)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title("Single letter: " + single_letter)
    
    ax.set_xticks(np.array([0,1/4,1/2,3/4,1])*letter_width_l,  labels=["0","","","",letter_width_l])
    ax.set_yticks(np.array([0,1/4,1/2,3/4,1])*letter_height_l, labels=["0","","","",letter_height_l])
    ax.grid(True)
    
    # Print one single letter
    
    printSentence(ax, 
                  single_letter, 
                  points_per_unit_l, 
                  letter_width_l, 
                  letter_height_l, 
                  space_width_l, 
                  x_start, 
                  y_start, 
                  points_dispersion_l, 
                  points_size_l, 
                  points_cmap_l)
    
    # Image construction for the sentence
    
    sentence_xticks = np.arange(x_start+letter_width_s/2, x_start+(letter_width_s+space_width_s)*len(sentence)-space_width_s, letter_width_s+space_width_s) # xticks position
    
    fig, ax = plt.subplots(dpi=2000)
    ax.set_aspect('equal', 'box')
    
    ax.set_title("Sentence example")
    ax.set_xlim(-space_width_s, (letter_width_s+space_width_s)*len(sentence))
    
    plt.box(False)
    ax.set_yticks([])
    ax.set_xticks(sentence_xticks)
    ax.tick_params(length = 0)
    ax.set_xticklabels(np.arange(0,len(sentence_xticks))+1, fontsize='x-small')
    
    # Print the sentence
    
    printSentence(ax, 
                  sentence, 
                  points_per_unit_s, 
                  letter_width_s, 
                  letter_height_s, 
                  space_width_s, 
                  x_start, 
                  y_start, 
                  points_dispersion_s, 
                  points_size_s, 
                  points_cmap_s)