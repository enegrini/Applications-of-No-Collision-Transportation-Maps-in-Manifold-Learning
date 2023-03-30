#Import required packages
import numpy as np
import matplotlib.pyplot as plt
import ot
import ot.plot
import sys
import math
import os


#Function to compute nearest neighbors of generated points on the grid
def lookup(point_x,point_y,x,y):
    x_grid = np.abs(x-point_x).argmin()
    y_grid = np.abs(y-point_y).argmin()
    return [x_grid, y_grid]

def DiscreteGaussian(mu,cov,x,y,n,m):
    
    #Use in-built function of ot package to generate samples from reference distribution (could as well use np.random)
    ref_points = ot.datasets.make_2D_samples_gauss(m, mu, cov)
    
    #Map reference samples to closest grid points
    ref_grid = np.zeros((m,2), dtype=int)
    for i in range(0,m):
            tmp = lookup(ref_points[i][0],ref_points[i][1],x,y)
            ref_grid[i][0] = tmp[0]
            ref_grid[i][1] = tmp[1]

    ref_count = np.zeros((n,n), dtype=int)
    for i in range(0,m):
        ref_count[ref_grid[i][0],ref_grid[i][1]] = ref_count[ref_grid[i][0],ref_grid[i][1]] + 1

    #Compute the reference discrete distributions after mapping to grid points
    r = []
    for i in range(0,n):
        for j in range(0,n):
            r.append(ref_count[i][j]/ref_count.sum())
    r = np.asarray(r)
    
    return ref_points, r
