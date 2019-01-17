from __future__ import absolute_import
from __future__ import print_function

import cv2
import numpy as np
import itertools

from helper import *
import os

def normalized(rgb):
    #return rgb/255.0
    norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)

    b=rgb[:,:,0]
    g=rgb[:,:,1]
    r=rgb[:,:,2]

    norm[:,:,0]=cv2.equalizeHist(b)
    norm[:,:,1]=cv2.equalizeHist(g)
    norm[:,:,2]=cv2.equalizeHist(r)

    return norm

def one_hot_it(labels):
    dim1 = labels.shape[0]
    dim2 = labels.shape[1]
    x = np.zeros([dim1,dim2,4])
    for i in range(dim1):
        for j in range(dim2):
            label = labels[i][j]
            if label == 127:
                x[i,j,0]=1
            elif label == 191:
                x[i,j,1]=1
            elif label == 255:
                x[i,j,2]=1
            elif label==64:
                x[i,j,3]=1
            #elif label==0:
                #x[i,j,4]=1
    return x

def one_hot_reverse(label, color_label):
    dim1 = label.shape[0]
    dim2 = label.shape[1]
    x = np.zeros([dim1,dim2,3])
    for i in range(dim1):
        for j in range(dim2):
            if max(label[i, j])>0:
                class_x = label[i,j].argmax()
                x[i,j,:] = color_label[class_x,:]
    return x


def computeIoU(label,pred_label):
# label: targets of test data with size [nb_batches,d_shape,d_shape, nb_classes]
# pred_label: predicted targets, same size as label 
    nb_class = label.shape[3]
    nb_batch = label.shape[0]
    d_shape = label.shape[1]
    I = np.zeros([nb_class]) # intersection
    U = np.zeros([nb_class]) # union
    for n_img in range(nb_batch):
        for i_pix in range(d_shape):
            for j_pix in range(d_shape):
                pix_label = label[n_img][i_pix][j_pix].argmax()
                pix_label_pred = pred_label[n_img][i_pix][j_pix].argmax()
                if pix_label == pix_label_pred:
                    I[pix_label] += 1
                    U[pix_label] += 1
                else:
                    U[pix_label] += 1
                    U[pix_label_pred] += 1
    IoU = np.true_divide(I,U)

    return IoU

def colorToclass(label):
    x = np.zeros((label.shape[0],label.shape[1]))
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i,j] == 127:
                x[i,j]=0
            elif label[i,j] == 191:
                x[i,j]=1
            elif label[i,j] == 255:
                x[i,j]=2
            elif label[i,j]==64:
                x[i,j]=3
            elif label[i,j]==0:
                x[i,j]=4
    return x


def classTocolor(label):
    x = np.zeros((label.shape[0], label.shape[1]))
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i, j] == 0:
                x[i, j] = 127
            elif label[i, j] == 1:
                x[i, j] = 191
            elif label[i, j] == 2:
                x[i, j] = 255
            elif label[i, j] == 3:
                x[i, j] = 64
            elif label[i, j] == 4:
                x[i, j] = 0
    return x