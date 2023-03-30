#Import required packages
import numpy as np
import matplotlib.pyplot as plt
import ot
import ot.plot
import sys
import math
import os
import tensorflow
from tensorflow.keras.datasets import mnist
from scipy.linalg import eigh

def ImportImages(Class_X,Class_Y,img):
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

#     #shape of dataset
#     print('X_train: ' + str(train_X.shape))
#     print('Y_train: ' + str(train_y.shape))
#     print('X_test:  '  + str(test_X.shape))
#     print('Y_test:  '  + str(test_y.shape))

    num_train = train_X.shape[0]
    for j in range(num_train):
        if train_y[j] == img:
            Class_X.append(train_X[j])
            Class_Y.append(train_y[j])

def PlotSamples(Class_X,Class_Y):
    #Plot images
    num_row = 2
    num_col = 2
    num = 4
    # plot images
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
    for i in range(num):
        ax = axes[i//num_col, i%num_col]
        ax.imshow(Class_X[i], cmap=plt.get_cmap('gray'))
        ax.set_title('Label: {}'.format(Class_Y[i]))
    plt.tight_layout()
    plt.show()

# def ShearImages(Class_X,PointClouds,Masses,Pixels,n_shears,lambd_min,lambd_max,shft_min,shft_max):
#     shearer = Shear_LOT.Shear()
#     for i in range(0,n_shears):

#         shear_angle = np.random.uniform(0,360)
#         shft1 = np.random.randint(shft_min,shft_max)
#         shft2 = np.random.randint(shft_min,shft_max)
#         l1 = np.random.uniform(lambd_min,lambd_max)
#         l2 = np.random.uniform(lambd_min,lambd_max)
    
#         shearer.create_shear(angle=shear_angle,
#                      lambda_1=l1,
#                      lambda_2=l2,
#                      shift=[ [shft1], [shft2] ],
#                      center=[[13], [13]])
#         sheared_tmp = shearer.shear_image(image=Class_X[i])
#         sheared_tmp = np.asarray(sheared_tmp)

#         Pixels[i] = sheared_tmp.flatten()
#         pixel_sum = np.sum(Pixels[i])
#         Pixels[i] = (1/pixel_sum)*Pixels[i]
    
#         #Extract support of point cloud
#         idxes = np.where(sheared_tmp>0)
#         cloud = np.zeros((len(idxes[0]), 2))
#         cloud[:, 0] = idxes[1]/28
#         cloud[:, 1] = 1 - idxes[0]/28
#         PointClouds[i] = cloud
#         #Extract masses of point cloud
#         nu = sheared_tmp[np.where(sheared_tmp>0)] + 1
#         Masses[i] = nu / np.sum(nu)

# def PlotShears(Class_X,PointClouds,Pixels,n_shears,tot_im):

#     #Visualize the point clouds representation of a subset of the sheared ones
#     plt.figure(figsize=(20,6))
#     #print("Note that imshow (first two rows) interpolates pixel values. The third row scatter plots the points which have non-zero pixel values.")
    
#     for i in range(tot_im):
#         plt.subplot(3, tot_im, i+1)
#         idx = np.random.randint(0,n_shears)
#         # show original image
#         plt.imshow(Class_X[idx], cmap='Greys')
#         plt.axis('off')

#         plt.subplot(3, tot_im, tot_im+1+i)
#         plt.imshow(Pixels[idx].reshape(28,28), cmap='Greys')
#         plt.axis('off')

#         # show associated point cloud
#         plt.subplot(3, tot_im, 2*tot_im+1+i)
#         cloud = PointClouds[idx]
#         plt.scatter(cloud[:, 0], cloud[:, 1], s=1)
#         plt.xlim((-0.05, 1.05))
#         plt.ylim((-0.05, 1.05))
#         plt.axis('off')


def FullDiscreteEmbed(template,reference,Cost,grid_size):
    Embedding = []
    Support = []
    
    TransportPlan =  ot.lp.emd(reference,template,Cost)
#     print(TransportPlan.shape)
    for j in range(0,TransportPlan.shape[0]):
        if np.max(TransportPlan[j]) > 1e-12:
            x_index = math.floor(np.argmax(TransportPlan[j]) / grid_size)
            y_index = np.argmax(TransportPlan[j]) % grid_size      
            Embedding.append([x_index, y_index])
            Support.append(j)
    
    Embedding = np.asarray(Embedding)
    Support = np.asarray(Support)
    
    return Embedding, Support
