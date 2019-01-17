#from Tkinter import *
#from PIL import ImageTk, Image
#import ttk
import os

import matplotlib
matplotlib.use('Agg')

import math

import cv2
import numpy as np
from helper import *

# resultPath = '../Result/2805_11_C1_RS/'

resultPath = '../../time1/test_sofar/c617_on_c617/1204_13_bs_aug/'
conf_mat = np.load(resultPath + 'conf_mat_minus1.npy')
conf_mat = conf_mat[0:4, 0:4]

nb_class = conf_mat.shape[0]
recall = np.zeros((nb_class))
precision = np.zeros((nb_class))
IOU = np.zeros((nb_class))
tp_sum = 0

for class_i in range(nb_class):
    tp_i = conf_mat[class_i,class_i]
    recall[class_i] = tp_i/np.sum(conf_mat[class_i,:])
    precision[class_i] = tp_i/np.sum(conf_mat[:,class_i])
    tp_sum += tp_i
    IOU[class_i] = tp_i/(np.sum(conf_mat[class_i, :]) + np.sum(conf_mat[:, class_i]) - tp_i)

total_acc = tp_sum/np.sum(conf_mat)

mean_IOU = np.sum(IOU)/4

#------------------------------------------------------------------------------------------------------------------------#
# record the results in the log file
text_file = open(resultPath+'result_medianM1.txt',"w")

text_file.write('------------------------------------------------------------------------------')

text_file.write("\nConfusion Matrix:\n")
for i in range(nb_class):
    for j in range(nb_class):
        text_file.write("%15d"% conf_mat[i,j])
    text_file.write('\n')

text_file.write('\nRecall: ')
for j in range(nb_class):
    text_file.write("%f "% recall[j])

text_file.write('\n\nPrecision: ')
for j in range(nb_class):
    text_file.write('%f '% precision[j])

text_file.write('\n\nOverall accuracy: %f\n'%total_acc)

for j in range(nb_class):
    text_file.write('\n%f ' % IOU[j])
text_file.write('\n\nMean IoU: %f' % mean_IOU)
text_file.close()
