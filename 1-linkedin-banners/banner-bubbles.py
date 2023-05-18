#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 22:40:28 2023

@author: andreaboselli
"""

#%% Preparatory code

# Useful settings

image_dpi = 2000

right_limit = 18
upper_limit =  4.5

n_sampled_pois = 150

small_words = {
                "points_per_unit":   150,
                "letter_width":      0.1451,
                "letter_height":     0.25,
                "space_width":       0.0484,
                "points_dispersion": 1/120,
                "points_size":       0.18,
                "points_cmap":       'ocean',
              }


# Import libraries

import BubbleSentences   as bs
import numpy             as np
import matplotlib.pyplot as plt


# Define image shape

fig, ax = plt.subplots(dpi=image_dpi)
ax.set_aspect('equal', 'box')

ax.set_xlim(left   = 0, right = right_limit)
ax.set_ylim(bottom = 0, top   = upper_limit)

ax.set_xticks([])
ax.set_yticks([])

for axis in ['top', 'bottom', 'left', 'right']:

    ax.spines[axis].set_linewidth(0.1)
    ax.spines[axis].set_color('gray')
    
filled_blocks = []


#%% Bubbles for "ANDREA BOSELLI"

np.random.seed(2) # OK

curr_block = bs.printSentence(axis              = ax,
                              sentence          = "ANDREA BOSELLI",
                              points_per_unit   = 85,
                              letter_width      =  0.552,
                              letter_height     =  0.88,
                              space_width       =  0.184,
                              x_start           =  7.25,
                              y_start           =  2.33,
                              points_dispersion = 1/35,
                              points_size       =  1,
                              points_cmap       = 'ocean')
filled_blocks.append(curr_block)


#%% Bubbles for "MATHEMATICAL ENGINEER"

np.random.seed(3) # OK

curr_block = bs.printSentence(axis              = ax,
                              sentence          = "MATHEMATICAL ENGINEER",
                              points_per_unit   = 110,
                              letter_width      =   0.297,
                              letter_height     =   0.47,
                              space_width       =   0.0992,
                              x_start           =   7.25,
                              y_start           =   1.3,
                              points_dispersion = 1/60,
                              points_size       =   0.5,
                              points_cmap       = 'ocean')
filled_blocks.append(curr_block)


#%% Bubbles for "ENGINEERING"

np.random.seed(1) # 

curr_block = bs.printSentence(axis              = ax,
                              sentence          = "ENGINEERING",
                              points_per_unit   = small_words["points_per_unit"],
                              letter_width      = small_words["letter_width"],
                              letter_height     = small_words["letter_height"],
                              space_width       = small_words["space_width"],
                              x_start           = 0.37,
                              y_start           = 3.9,
                              points_dispersion = small_words["points_dispersion"],
                              points_size       = small_words["points_size"],
                              points_cmap       = small_words["points_cmap"])
filled_blocks.append(curr_block)


#%% Bubbles for "MATHEMATICS"

np.random.seed(1) # 

curr_block = bs.printSentence(axis              = ax,
                              sentence          = "MATHEMATICS",
                              points_per_unit   = small_words["points_per_unit"],
                              letter_width      = small_words["letter_width"],
                              letter_height     = small_words["letter_height"],
                              space_width       = small_words["space_width"],
                              x_start           = 0.74,
                              y_start           = 2.78,
                              points_dispersion = small_words["points_dispersion"],
                              points_size       = small_words["points_size"],
                              points_cmap       = small_words["points_cmap"])
filled_blocks.append(curr_block)


#%% Bubbles for "PROGRAMMING"

np.random.seed(1) # 

curr_block = bs.printSentence(axis              = ax,
                              sentence          = "PROGRAMMING",
                              points_per_unit   = small_words["points_per_unit"],
                              letter_width      = small_words["letter_width"],
                              letter_height     = small_words["letter_height"],
                              space_width       = small_words["space_width"],
                              x_start           = 3.92,
                              y_start           = 3.41,
                              points_dispersion = small_words["points_dispersion"],
                              points_size       = small_words["points_size"],
                              points_cmap       = small_words["points_cmap"])
filled_blocks.append(curr_block)


#%% Bubbles for "STATISTICS"

np.random.seed(1) # 

curr_block = bs.printSentence(axis              = ax,
                              sentence          = "STATISTICS",
                              points_per_unit   = small_words["points_per_unit"],
                              letter_width      = small_words["letter_width"],
                              letter_height     = small_words["letter_height"],
                              space_width       = small_words["space_width"],
                              x_start           = 4.3,
                              y_start           = 1.78,
                              points_dispersion = small_words["points_dispersion"],
                              points_size       = small_words["points_size"],
                              points_cmap       = small_words["points_cmap"])
filled_blocks.append(curr_block)


#%% Bubbles for "DATA SCIENCE"

np.random.seed(1) # 

curr_block = bs.printSentence(axis              = ax,
                              sentence          = "DATA SCIENCE",
                              points_per_unit   = small_words["points_per_unit"],
                              letter_width      = small_words["letter_width"],
                              letter_height     = small_words["letter_height"],
                              space_width       = small_words["space_width"],
                              x_start           = 4.53,
                              y_start           = 0.5,
                              points_dispersion = small_words["points_dispersion"],
                              points_size       = small_words["points_size"],
                              points_cmap       = small_words["points_cmap"])
filled_blocks.append(curr_block)


#%% Additional pois

np.random.seed(1)


# Sample n_sampled_pois pois outside the texts boxes

filled_blocks = np.array(filled_blocks) # cast the text boxes coordinates as a NumPy array

sampled_pois = []                       # list of sampled pois
pois_counter = 0                        # counter of sampled pois

while(pois_counter < n_sampled_pois):
    
    # Sample x and y coordinates
    
    curr_x = np.random.rand()*right_limit
    curr_y = np.random.rand()*upper_limit
    
    # Check that the current point outside all boxes

    outside_boxes = np.logical_not( np.any( (curr_x > filled_blocks[:,0]) & (curr_x < filled_blocks[:,1]) & (curr_y > filled_blocks[:,2]) & (curr_y < filled_blocks[:,3]) ) )
    
    # If the point is outside the text boxes, save it and increase the counter
    
    if(outside_boxes):
        sampled_pois.append([curr_x,curr_y])
        pois_counter += 1

sampled_pois = np.array(sampled_pois)


# Set color and dimension of the pois

pois_c    = np.random.randint(0, n_sampled_pois, n_sampled_pois)
pois_s    = np.abs(np.random.randn(n_sampled_pois))*2
pois_cmap = 'ocean'
alpha     = 0.8


# Plot the pois

ax.scatter(x          = sampled_pois[:,0], 
           y          = sampled_pois[:,1], 
           c          = pois_c,
           edgecolors = 'none',
           s          = pois_s,
           cmap       = pois_cmap,
           alpha      = alpha)