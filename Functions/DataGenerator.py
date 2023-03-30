## Import statements

import sklearn
import numpy as np
import ot
import ot.plot

# networkx is a graph library 
import networkx as nx

import time

from sklearn import manifold as man
from sklearn.decomposition import PCA
from pydiffmap import diffusion_map as dm

import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.sparse
from pydiffmap import diffusion_map as dm

from torch import HalfStorage

# NOTE: RBFInterpolator needs scipy>=1.7.0
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import LinearNDInterpolator
#from scipy.interpolate import RBFInterpolator

import math



from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import LinearNDInterpolator



## Voxels vs. Pointclouds

# There are two representations of images that are used in this notebook. Wassmap considers an image as a "pointcloud", i.e., as a set of coordinates (x,y,w) where (x,y) corresponds to the 2-d spatial location and w is the pixel intensity at that location. This is representing a dirac train of the form $$\sum w_{ij}\delta_{(x_i,y_j)}$$
# where $(x_i,y_j)$ come from a prescribed meshgrid based on the voxel image size.
#All the other methods consider an image as a "voxel array", i.e., as a matrix whose entries are the pixel intensity values, typically either between 0 and 1 or integers between 0 and 255 for greyscale images.

# The following functions allow one to pass back and forth from one representation to another.
# Functions for voxel reps and pointcloud reps
# NOTE: classical Isomap is defined for voxelized images, not pointclouds.

def vox_to_pointcloud(voxarray,grid,eps=0):
    # Convert a voxel representation ('voxarray') to a (weighted) point cloud representation 
    # Assume that the input grid is in "ij meshgrid" format i.e. the grid has two layers, xx and yy, each size (nx)-by-(ny)
    # The voxel array will be unrolled via "column-major" order ("Fortran/Matlab" ordering)
    # Note that initially the number of voxels must equal the number of grid points - but 
    # zero voxels will be removed from the representation (i.e. no points with weight zero allowed)
    # if the optional parameter eps is passed, voxels with value less or equal to eps will be dropped
    # The returned array consists of (x,y,w) tuples i.e. X = [x1,y1,w1;x2,y2,w2;...;xP,yP,wP] where P 
    # is the number of nonzero points 
    xx,yy = np.squeeze(np.split(grid,2))
    X = np.vstack((xx.ravel(),yy.ravel())).T
    nX  = X.shape[0] # Number of points = number of rows
    nvi = voxarray.shape[0] # Number of voxel rows
    nvj = voxarray.shape[1] # Number of voxel cols 
    if nX != nvi*nvj: raise ValueError("Number of grid points must equal number of voxels!")
    X = np.concatenate((X,voxarray.T.reshape(nvi*nvj,1)),axis=1)
    return X[X[:,-1]>eps,:]

def pointcloud_to_vox(array,grid):
    # Converts a pointcloud representation to a voxel representation 
    # Assumes that the grid is in "ij meshgrid" format i.e. grid has two layers, xx and yy; each are size (nx+1)-by-(ny+1)
    # The grid points are assumed to define the corners of the voxels, so the 
    # voxel rep will be a single nx-by-ny array with entries equal to the average pointcloud weights
    # V_ij = \mean_k W_k if (x_k,y_k) is in voxel ij
    xx,yy = np.squeeze(np.split(grid,2))
    points = array[:,0:2]
    values = array[:,2]
    # One can use different interpolators here. NearestNDInterpolator uses nearest neighbor values to fill in missing
    # grid entries, but this results in a coarser image. LinearND is smoother.
#   interp = NearestNDInterpolator(points,values)
    interp = LinearNDInterpolator(points,values,fill_value=0.0)
    X = interp(xx,yy).T
    return X


## Synthetic Image Generation
# The functions below generate pointcloud-format images corresponding to indicator functions of certain domains in $\mathbb{R}^2$. One can generate rectangles, triangles, circles, and ellipses. Standard usage of these functions may be found in the Experiments later in the notebook.


def generate_rectangle(side0, side1, initial_point=[0,0], samples=100):
    # Generates a rectangle in point cloud format
    x = np.linspace(initial_point[0], initial_point[0]+side0, num=samples)
    y = np.linspace(initial_point[1], initial_point[1]+side1, num=samples)
    return np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])

def in_triangle(endpoint1,endpoint2,endpoint3,point):
    # Indicator function of a triangle 
    # Returns 1 (True) if point is in the triangle, zero (False) else
    c1 = (endpoint2[0]-endpoint1[0])*(point[1]-endpoint1[1]) - (endpoint2[1]-endpoint1[1])*(point[0]-endpoint1[0])
    c2 = (endpoint3[0]-endpoint2[0])*(point[1]-endpoint2[1]) - (endpoint3[1]-endpoint2[1])*(point[0]-endpoint2[0])
    c3 = (endpoint1[0]-endpoint3[0])*(point[1]-endpoint3[1]) - (endpoint1[1]-endpoint3[1])*(point[0]-endpoint3[0])

    if (c1<0 and c2<0 and c3<0) or (c1>0 and c2>0 and c3>0):
        return True
    else:
        return False    

def generate_triangle(endpoint1, endpoint2, endpoint3, samples=100):
    # Generates a triangle in point cloud format
    x = np.linspace(min(endpoint1[0],endpoint2[0],endpoint3[0]), max(endpoint1[0],endpoint2[0],endpoint3[0]), num=samples)
    y = np.linspace(min(endpoint1[1],endpoint2[1],endpoint3[1]), max(endpoint1[1],endpoint2[1],endpoint3[1]), num=samples)
    xy_0 = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
    xy = []
    for point in xy_0:
        if in_triangle(endpoint1,endpoint2,endpoint3,point):
            xy.append(point)
    return np.array(xy)

def in_circle(center, radius, point):
    # Indicator function of a circle
    # Returns 1 (True) if point is in the circle, zero (False) else
    if (point[1]-center[1])**2+(point[0]-center[0])**2<=radius**2:
        return True
    else:
        return False

def generate_circle(center, radius, samples=100):
    # Generates a circle in pointcloud format
    x = np.linspace(center[0]-radius, center[0]+radius, num=samples)
    y = np.linspace(center[1]-radius, center[1]+radius, num=samples)
    xy_0 = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
    xy = []
    for point in xy_0:
        if in_circle(center,radius,point):
            xy.append(point)
    return np.array(xy)

def generate_ellipse(center, axis_x, axis_y,samples = 100):
    # Requires running of transformation cell below
    # Generates an ellipse in pointcloud format by dilating a circle
    circle = generate_circle([0,0],1,samples)
    ellipse = dilation(circle,[axis_x, axis_y])
    ellipse = translation(ellipse,center)
    return np.array(ellipse)


## Transformations of Images
# Here we implement various operations on an input image: rotation, translation, and dilation. Note that dilations are along the coordinate axes and of the form $f(\text{diag}(\theta_0,\theta_1)x)$, but if one wants to do generic dilation one can rotate and then dilate an image. These functions should work on both voxel and pointcloud images.

def rotation(object, radian_degree):
    # Rotates an object counterclockwise by radian_degree
    A = [[math.cos(radian_degree), -math.sin(radian_degree)],[math.sin(radian_degree), math.cos(radian_degree)]]
    image = []
    for index,point in enumerate(object):
        image.append(np.matmul(A,point))
    return np.array(image)

def translation(object, translate_direction):
    # translate_direction is a list or ndarray of the form [t, s]
    # function translates an object located at [x,y] to [x+t, y+s]
    object_array = np.array(object)
    direction_array = np.array(translate_direction)
    image = [x + direction_array for x in object_array]
    return np.array(image)

def dilation(object, parameter):
    # Dilates an object by multiplying it by a diagonal matrix
    # parameter is a list or ndarray of size 2
    A = [[parameter[0], 0],[0, parameter[1]]]
    image = []
    for index,point in enumerate(object):
        image.append(np.matmul(A,point))
    return np.array(image)

#Distance Function and Error Norms
def pairwiseDist(features):
    distance = np.zeros((len(features),len(features)))
    
    for i in range(len(features)):
        for j in range(i+1,len(features)):
            distance[i,j] = np.linalg.norm(features[i]-features[j])**2
    distance += distance.T
    return distance

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def relErr(TrueD, ApproxD):
    return np.round(100*np.linalg.norm(TrueD-ApproxD)/np.linalg.norm(TrueD),4)
