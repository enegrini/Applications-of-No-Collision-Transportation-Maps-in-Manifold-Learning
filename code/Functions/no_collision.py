import scipy.ndimage as nd
from math import *
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
from skimage.transform import resize
from PIL import Image
import scipy.io

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import datasets
import numpy as np
from mpl_toolkits import mplot3d
from keras.datasets import mnist

from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def normalize(img):
    img = (img-img.min())/(img.max()-img.min())*(1-2.5e-16) +2.5e-16 #makes the min or image 2.5e-16
    return img

def makePdensity(img):
    img = normalize(img) #add contant density everywhere
    probimg = img/img.sum()
    return probimg


# Given an image it cuts it so that the two halves have density (img density/2).
# plots image on original axes so indices of cut are in the original coordinates

def half_cutID_CM(Pimage, eps, cut ='V'):
    if cut == 'V':
        axis = 0
    elif cut == 'H':
        axis = 1
    else:
        print('incorrect cut direction')
    project_x = np.sum(Pimage, axis= axis)
    #find column index whwre half density is reached (computed as img density/2, may not be exact)
    sum_x = 0
    index_half = -1
    for el in project_x:
        sum_x += el
        index_half += 1
        if sum_x >= np.sum(project_x)/2:
            break
    #define half image
    if cut == 'V':
        halfdensyL = np.copy(Pimage)
        halfdensyL[:,index_half:] = np.zeros(halfdensyL[:,index_half:].shape) + eps

        
        halfdensyR = np.copy(Pimage)
        halfdensyR[:,:index_half] = np.zeros(halfdensyR[:,:index_half].shape) + eps

    elif cut == 'H':
        halfdensyL = np.copy(Pimage)
        halfdensyL[index_half:,:] = np.zeros(halfdensyL[index_half:,:].shape) + eps

        
        halfdensyR = np.copy(Pimage)
        halfdensyR[:index_half,:] = np.zeros(halfdensyR[:index_half,:].shape) + eps

        
    else:
        print('incorrect cut direction')
    
    return halfdensyL,halfdensyR, index_half, cut


# # Given image and number of cuts, recursively cuts image and plots cuts, starting cut direction can be defined, default vertical
# # Also computes center of mass of cuts

def rCutsV_CM(Pimage,N, eps , cut = 'V'):
    if N==0:
        projx = np.sum(Pimage, axis= 0)
        ind_x = np.array([i for i in range(len(projx))])
        projy = np.sum(Pimage, axis= 1)
        ind_y = np.array([i for i in range(len(projy))])
        sum_projx = np.sum(projx)
        sum_projy = np.sum(projy)
        # set coords to origin if cut is empty (maybe it's better to use geom. center)
        if sum_projx < eps*Pimage.size + 2*eps or sum_projy < eps*Pimage.size + 2*eps:
#             print('there are zeros')
#             x_coordM = np.sum(np.multiply(ind_x, projx))/np.sum(projx)
#             y_coordM = np.sum(np.multiply(ind_y, projy))/np.sum(projy)
            x_coordM = None
            y_coordM = None
        else:
            x_coordM = np.sum(np.multiply(ind_x, projx))/sum_projx
            y_coordM = np.sum(np.multiply(ind_y, projy))/sum_projy
#         plt.figure()
#         plt.imshow(Pimage)       
#         plt.plot(x_coordM, y_coordM, "or", markersize=4, label = 'center of mass')
#         plt.legend()
#         plt.title('Sum '+ '{:2e}'.format(Pimage.sum()),fontsize = 15)
        return (x_coordM,y_coordM)
    else:
        halfdensyL, halfdensyR, index_half, cut = half_cutID_CM(Pimage,eps, cut = cut)
#         print(halfdensyL.shape, halfdensyL.min(), halfdensyR.shape, halfdensyR.min())
#         print('N',N,'cut direction', cut, 'index of cut', index_half)
        if cut =='V':
            cut = 'H'
        else: 
            cut = 'V'
        return rCutsV_CM(halfdensyL,N-1,eps, cut = cut), rCutsV_CM(halfdensyR,N-1,eps, cut = cut)

# Given an image it cuts it so that the two halves have density (img density/2).
# plots image on original axes so indices of cut are in the original coordinates
# returns also coordinates of cuts

def half_cutID_GC(Pimage,eps, which_half = 'L', coo_cutL = [],coo_cutR = [], cut ='V'):
    if len(coo_cutL)==0:
        coo_cutL = [0, Pimage.shape[0]-1, 0, Pimage.shape[1]-1]
    if len(coo_cutR)==0:
        coo_cutR = [0, Pimage.shape[0]-1, 0, Pimage.shape[1]-1]
    if cut == 'V':
        axis = 0
    elif cut == 'H':
        axis = 1
    else:
        print('incorrect cut direction')
    project_x = np.sum(Pimage, axis= axis)
    #find column index whwre half density is reached (computed as img density/2, may not be exact)
    sum_x = 0
    index_half = -1
    for el in project_x:
        sum_x += el
        index_half += 1
        if sum_x >= np.sum(project_x)/2:
            break
    #define half image
    if cut == 'V':
        halfdensyL = np.copy(Pimage)
        halfdensyL[:,index_half:] = np.zeros(halfdensyL[:,index_half:].shape)+ eps
        halfdensyR = np.copy(Pimage)
        halfdensyR[:,:index_half] = np.zeros(halfdensyR[:,:index_half].shape)+ eps

        if which_half == 'L':
            coo_cutLL = np.copy(coo_cutL)
            coo_cutLR = np.copy(coo_cutR)
            coo_cutLL[0] = coo_cutL[0]
            coo_cutLL[1] = coo_cutL[1]
            coo_cutLL[2] = coo_cutL[2]
            coo_cutLL[3] = index_half
            
            coo_cutLR[0] = coo_cutL[0]
            coo_cutLR[1] = coo_cutL[1]
            coo_cutLR[2] = index_half
            coo_cutLR[3] = coo_cutL[3]
            
            cutL = coo_cutLL
            cutR = coo_cutLR
            
        else:
            coo_cutRL = np.copy(coo_cutL)
            coo_cutRR = np.copy(coo_cutR)
            coo_cutRL[0] = coo_cutR[0]
            coo_cutRL[1] = coo_cutR[1]
            coo_cutRL[2] = coo_cutR[2]
            coo_cutRL[3] = index_half
            
            coo_cutRR[0] = coo_cutR[0]
            coo_cutRR[1] = coo_cutR[1]
            coo_cutRR[2] = index_half
            coo_cutRR[3] = coo_cutR[3]
            
            cutL = coo_cutRL
            cutR = coo_cutRR
        
    elif cut == 'H':
        halfdensyL = np.copy(Pimage)
        halfdensyL[index_half:,:] = np.zeros(halfdensyL[index_half:,:].shape)+ eps
        halfdensyR = np.copy(Pimage)
        halfdensyR[:index_half,:] = np.zeros(halfdensyR[:index_half,:].shape)+ eps
        
        if which_half == 'L':            
            coo_cutLL = np.copy(coo_cutL)
            coo_cutLR = np.copy(coo_cutR)
            coo_cutLL[0] = coo_cutL[0]
            coo_cutLL[1] = index_half
            coo_cutLL[2] = coo_cutL[2]
            coo_cutLL[3] = coo_cutL[3]
            
            coo_cutLR[0] = index_half
            coo_cutLR[1] = coo_cutL[1]
            coo_cutLR[2] = coo_cutL[2]
            coo_cutLR[3] = coo_cutL[3]
            
            cutL = coo_cutLL
            cutR = coo_cutLR
        else:
            coo_cutRL = np.copy(coo_cutL)
            coo_cutRR = np.copy(coo_cutR)
            coo_cutRL[0] = coo_cutR[0]
            coo_cutRL[1] = index_half
            coo_cutRL[2] = coo_cutR[2]
            coo_cutRL[3] = coo_cutR[3]
            
            coo_cutRR[0] = index_half
            coo_cutRR[1] = coo_cutR[1]
            coo_cutRR[2] = coo_cutR[2]
            coo_cutRR[3] = coo_cutR[3]
            
            cutL = coo_cutRL
            cutR = coo_cutRR
        
        
    else:
        print('incorrect cut direction')
    
    return halfdensyL, halfdensyR, cutL, cutR, index_half, cut

# # Given image and number of cuts, recursively cuts image and plots cuts, starting cut direction can be defined, default vertical
# # Also computes geometric center and coordinates of cuts

def rCutsV_GC(Pimage,N,eps,  which_half = 'L',cutL = [], cutR = [],cut = 'V'):
    if N==0:
        if len(cutL)==0:
            cutL = [0, Pimage.shape[0]-1, 0, Pimage.shape[1]-1]
        if len(cutR)==0:
            cutR = [0, Pimage.shape[0]-1, 0, Pimage.shape[1]-1]
#         plt.figure()
#         plt.imshow(Pimage)  
#         if which_half == 'L':
#             plt.plot(GCxL, GCyL, "or", markersize=4, label = 'geometric center')
#         else:
#             plt.plot(GCxR, GCyR, "or", markersize=4, label = 'geometric center')
#         plt.legend()
#         plt.title('Sum '+str(np.round(Pimage.sum(),3)),fontsize = 15)
        if which_half == 'L':
            return  cutL
        else:
            return  cutR
    else:
        halfdensyL, halfdensyR,cutL, cutR, index_half, cut = half_cutID_GC(Pimage,eps, which_half = which_half, coo_cutL = cutL, coo_cutR = cutR, cut = cut)
#         print('N',N,'cut direction', cut, 'index of cut', index_half)
        if cut =='V':
            cut = 'H'
        else: 
            cut = 'V'
        return rCutsV_GC(halfdensyL,N-1, eps, which_half = 'L', cutL = cutL, cutR = cutR, cut = cut), rCutsV_GC(halfdensyR,N-1,eps,which_half = 'R', cutL = cutL, cutR = cutR, cut = cut)

#makes list containing coordinates of centers    
def centers_list(coords_tuple):
    Coord_arr = np.array(coords_tuple).flatten()
    CoordsList = []
    for i in range(0,len(Coord_arr),2):
        CoordsList += [[Coord_arr[i], Coord_arr[i+1]]]
    return np.array(CoordsList)


# other functions for plots

def xy_lists(coords_tuple):
    Coord_arr = np.array(coords_tuple).flatten()
    x = [Coord_arr[i] for i in range(0,len(Coord_arr),2)]
    y = [Coord_arr[i] for i in range(1,len(Coord_arr),2)]
    return np.array(x), np.array(y)

def cuts_list(cuts_tuple):
    cuts_array = np.array(cuts_tuple).flatten()
    cutsList = []
    for i in range(0,len(cuts_array),4):
        cutsList += [(cuts_array[i], cuts_array[i+1],cuts_array[i+2],cuts_array[i+3])]
    return cutsList

def GC_lists(cuts_coords):
    x = []
    y = []
    for tup in cuts_coords:
        GCy= (tup[1]-tup[0])/2 + tup[0]
        GCx = (tup[3]-tup[2])/2 + tup[2]
        x += [GCx]
        y += [GCy]
    return np.array(x),np.array(y)

def features_mat(data, N):
    feat_matCM = np.zeros((data.shape[0], 2**N*2))
    feat_matGC = np.zeros((data.shape[0], 2**N*2))
    for i in range(len(data)):
        img = data[i]
        probimg = makePdensity(img)
        eps = probimg.min()
        xG,yG = GC_lists(cuts_list(rCutsV_GC(probimg, N,eps)))
        xM, yM = xy_lists(rCutsV_CM(probimg,N,eps))
        xM[np.where(xM==None)] = xG[np.where(xM==None)]
        yM[np.where(yM==None)] = yG[np.where(yM==None)]
        feat_matCM[i] = np.concatenate((xM.reshape(-1,1),yM.reshape(-1,1)), axis=1).reshape(-1)
        feat_matGC[i] = np.concatenate((xG.reshape(-1,1),yG.reshape(-1,1)), axis=1).reshape(-1)
    return feat_matCM, feat_matGC
