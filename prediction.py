from __future__ import absolute_import
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import keras.models as models
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras import backend as K
from keras import callbacks
import math
import cv2
import numpy as np
import json
import h5py

from helper import *

K.set_image_dim_ordering('tf')  # channel_last data format

from data_loader import DataLoader
from dataGenerator import DataGenerator, get_crop
from modelTiramisu import Tiramisu


def prediction(result_path='_', model_name='C1_17_RS', pred_file='C1_17_RS.hdf5'):

    # --------------------------------------------------------------------------------------------------------------- #
    # Prediction on test set

    # Read HDF5 file
    hdf5_file = h5py.File('../Data/'+pred_file, 'r')

    dim_patch = 56
    num_fusion = 0

    params_test = {
        'hdf5_file': hdf5_file,
        'dim_x': dim_patch,
        'dim_y': dim_patch,
        'dim_z': 3,
        'num_fusion': num_fusion,
        'tag': 'Test'
    }

    test_setting = DataGenerator(**params_test)

    # get crop list, mean, and other parameters
    test_crop_list = test_setting.crop_list
    test_crop_per_im = test_setting.num_crop_per_im  # number of cropped patches per image

    overlap = 28

    print(test_crop_list[0])
    test_mean = test_setting.mean
    num_val = len(test_crop_list)/1809

    r_range = int(math.floor(test_setting.r/(dim_patch-overlap)))-1
    c_range = int(math.floor(test_setting.c/(dim_patch-overlap)))-1
    dim1_ = 784  # r_range*(dim_patch - overlap) + overlap  # actual size of the input & output image
    dim2_ = 1904  # c_range*(dim_patch - overlap) + overlap

    color_label = np.array([
        [127, 127, 127],  # water
        [191, 191, 191],  # ice
        [255, 255, 255],  # snow
        [64, 64, 64]  # clutter
    ])  # [0,0,0]

    # create a new folder
    pred_path = result_path + 'prediction/'
    prob_path = result_path + 'reliability/'

    try:
        os.makedirs(pred_path)
    except OSError:
        if not os.path.isdir(pred_path):
            raise

    try:
        os.makedirs(prob_path)
    except OSError:
        if not os.path.isdir(prob_path):
            raise

    # ------------------------------------------------------------------------------------------------------------------ #
    # load the model (and weights):
    # Tiramisu().create([None, None], [4, 6, 8], 10, [8, 6, 4], 12, 0.0001, 0.5, 5, model_name)
    load_model_name = '../Model/tiramisu_fc_dense' + model_name +'.json'
    model_file = open(load_model_name,'r')
    tiramisu = models.model_from_json(model_file.read())
    model_file.close()

    # load final weight
    tiramisu.load_weights(result_path + '/prop_tiramisu_weights_'+model_name+'.best.hdf5')

    # -------------------------------------------------------------------------------------------------------------------#
    # storing 3d prediction
    if num_fusion != 0:
        hdf5_pred = h5py.File(result_path + 'Pred.hdf5', 'w')
        hdf5_pred.create_dataset('pred', (num_val * 3, dim1_, dim2_, 4), np.float32)
        hdf5_pred.create_dataset('frame_name', (num_val * 3,), 'S18')

    # -------------------------------------------------------------------------------------------- #
    conf_mat = np.zeros((5, 5))
    im_name_list = hdf5_file["im_name"].value
    set_index_list = hdf5_file["set_index"].value

    for i in range(0, len(test_crop_list)/test_crop_per_im):
        i_im_crop = test_crop_list[i*test_crop_per_im][0][0:15]  # the name of the image the cropped patch belongs to
        print(i_im_crop)
        im_0_hdf5 = np.where(im_name_list==i_im_crop)[0][0]
        i_period = set_index_list[im_0_hdf5][1]
        i_im = set_index_list[im_0_hdf5][2]

        # ind0_t_file = t_file_list[1].index(i_im_crop)

        im = hdf5_file["im"][im_0_hdf5]
        conf_temp = np.zeros((5, 5))
        if num_fusion == 0:
            pred_im = np.zeros((dim1_, dim2_, 4))
        else:
            pred_im = np.zeros((2*num_fusion+1, dim1_, dim2_, 4))

        for i_crop in range(test_crop_per_im):
            if num_fusion != 0:
                X = np.zeros((num_fusion*2+1, dim_patch, dim_patch, 3))
            r_start, c_start = get_crop(dim_patch, dim_patch, test_setting.c, overlap, i_crop)

            #if num_fusion == 0:
            patch = im[r_start:r_start+dim_patch, c_start:c_start+dim_patch, :]
            X = patch - test_mean
            # else:
            #     im_name_stack = []
            #     for i_fuse in range(2*num_fusion + 1):
            #         # im_i_hdf5 = np.where(im_name_list==data_set_list.test[i_period][i_im+i_fuse-num_fusion])[0][0]
            #         ind_t_file = t_file_list[1].index(i_im_crop)-2*num_fusion + i_fuse
            #         im_i_hdf5 = np.where(im_name_list == t_file_list[1][ind_t_file])[0][0]
            #         patch = hdf5_file["im"][im_i_hdf5][r_start:r_start+dim_patch, c_start:c_start+dim_patch, :]
            #         X[i_fuse, :, :, :] = patch- test_mean
            #         im_name_stack.append(im_name_list[im_i_hdf5])

            # predict on batch (batch size 1)
            X = np.expand_dims(X, 0)
            pred_label = tiramisu.predict_on_batch(X)
            if num_fusion == 0:
                pred_im[r_start:r_start+dim_patch, c_start:c_start+dim_patch, :] += pred_label[0, :, :, :]
            else:
                pred_im[:, r_start:r_start+dim_patch, c_start:c_start+dim_patch, :] += pred_label[0, :, :, :, :]

        # if num_fusion == 0:
        label_image = pred_im/np.expand_dims(np.sum(pred_im, axis=-1), axis=-1)
        cv2.imwrite(pred_path+i_im_crop+'.png', one_hot_reverse(label_image, color_label))
        prob_image = np.amax(label_image, axis=-1)*255
        cv2.imwrite(prob_path+i_im_crop+'.png', prob_image)
        # else:
        #     for i_fuse in range(2*num_fusion+1):
        #         label_image = pred_im[i_fuse, :, :, :]/np.expand_dims(np.sum(pred_im[i_fuse, :, :, :], axis=-1), axis=-1)
        #         hdf5_pred["pred"][i*(2*num_fusion + 1) + i_fuse] = label_image
        #         hdf5_pred["frame_name"][i*(2*num_fusion + 1) + i_fuse] = im_name_stack[i_fuse]+'_'+str(i_fuse)
                # cv2.imwrite(pred_path+im_name_stack[i_fuse]+'_'+str(i_fuse)+ '.png', one_hot_reverse(label_image, color_label))
            # cv2.imwrite(pred_path + im_name_stack[i_fuse] + '_' + str(i_fuse) + '.png', one_hot_reverse(label_image, color_label))
            # prob_image = np.amax(label_image, axis=-1)*255
            # cv2.imwrite(prob_path+im_name_stack[i_fuse]+'_'+str(i_fuse)+'.png', prob_image)

        # evaluation (confusion matrix)
        id_pred = np.argmax(label_image, axis=-1)
        annot_img = hdf5_file['label'][im_0_hdf5]
        id_test = colorToclass(annot_img)

        for r in range(dim1_):
            for c in range(dim2_):
                conf_temp[int(id_test[r, c]), id_pred[r, c]] += 1
        conf_mat += conf_temp

    print(conf_mat)
    np.save(result_path + 'conf_mat_JDall', conf_mat)

    # -------------------------------------------------------------------------------------------------------------------#
    # precision, recall, overall accuracy, IOU
    conf_mat = conf_mat[0:4, 0:4]

    nb_class = conf_mat.shape[0]
    recall = np.zeros((nb_class))
    precision = np.zeros((nb_class))
    IOU = np.zeros((nb_class))
    tp_sum = 0

    for class_i in range(nb_class):
        tp_i = conf_mat[class_i, class_i]
        recall[class_i] = tp_i / np.sum(conf_mat[class_i, :])
        precision[class_i] = tp_i / np.sum(conf_mat[:, class_i])
        tp_sum += tp_i
        IOU[class_i] = tp_i / (np.sum(conf_mat[class_i, :]) + np.sum(conf_mat[:, class_i]) - tp_i)

    total_acc = tp_sum / np.sum(conf_mat)

    mean_IOU = np.sum(IOU) / 4

    # ---------------------------------------------------------------------------------------------------------------------#
    # record the results in the log file
    text_file = open(result_path + 'resultJDall.txt', "w")

    text_file.write('------------------------------------------------------------------------------')

    text_file.write("\nConfusion Matrix:\n")
    for i in range(nb_class):
        for j in range(nb_class):
            text_file.write("%15d" % conf_mat[i, j])
        text_file.write('\n')

    text_file.write('\nRecall: ')
    for j in range(nb_class):
        text_file.write("%f " % recall[j])

    text_file.write('\n\nPrecision: ')
    for j in range(nb_class):
        text_file.write('%f ' % precision[j])

    text_file.write('\n\nOverall accuracy: %f\n' % total_acc)

    for j in range(nb_class):
        text_file.write('\n%f ' % IOU[j])
    text_file.write('\n\nMean IoU: %f' % mean_IOU)
    text_file.close()


if __name__=='__main__':
    prediction()
