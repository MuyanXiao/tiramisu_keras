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
from modelTiramisu import Tiramisu

from helper import *
import argparse

K.set_image_dim_ordering('tf')  # channel_last data format

from dataGenerator import DataGenerator, get_crop


# parse parameters
parser = argparse.ArgumentParser(description="Test the model")
parser.add_argument("--result_dir", metavar="RESULT_DIR", type=str, default="../Result/", help="Path to store the predictions")
parser.add_argument("--model_name", metavar="MODEL_NAME", default="", type=str, help="Name specified for the model, e.g. C1_17_RS for prop_tiramisu_weights_C1_17_RS.best.hdf5")
parser.add_argument("--hdf5_file", metavar="HDF5_NAME", default="", type=str, help="Name of the hdf5 file")
parser.add_argument("--dim_patch", metavar="DIM_PATCH", default=224, type=int, help="size of the cropped patches, 56 for c0, 224 for c1")

# ------------------------------------------------------------------------------------------------------------------- #
# Prediction on test set
# ------------------------------------------------------------------------------------------------------------------- #
# prediction settings
def prediction(result_path='_', model_name='C0_17_RS', pred_file='C0_17_RS.hdf5', dim_patch=224):
    resultPath = result_path  # maybe change later to enter the path in the terminal

    # Read HDF5 file
    hdf5_file = h5py.File('../Data/' + pred_file, 'r')

    dim_patch = dim_patch
    num_fusion = 0  # the number of previous/following patch to be fused in the input

    params_test = {
        'hdf5_file': hdf5_file,
        'dim_x': dim_patch,
        'dim_y': dim_patch,
        'dim_z': 3,
        'num_fusion': num_fusion,
        'tag': 'Test'
        # 'mean': mean
    }

    test_setting = DataGenerator(**params_test)
    test_set_list = test_setting.set_list

    # get crop list, mean, and other parameters
    test_crop_list = test_setting.crop_list
    test_crop_per_im = test_setting.num_crop_per_im  # number of cropped patches per image

    overlap = dim_patch/2

    # print(test_crop_list[0])
    test_mean = test_setting.mean
    num_test = sum([len(i) for i in test_set_list])
    print(num_test)

    r_range = int(math.floor(test_setting.r/(dim_patch-overlap)))-1
    c_range = int(math.floor(test_setting.c/(dim_patch-overlap)))-1
    dim1_ = r_range*(dim_patch - overlap) + overlap  # actual size of the input & output image
    dim2_ = c_range*(dim_patch - overlap) + overlap

    color_label = np.array([
        [127, 127, 127],  # water
        [191, 191, 191],  # ice
        [255, 255, 255],  # snow
        [64, 64, 64]  # clutter
    ])  # [0,0,0]

    # create a new folder
    pred_path = resultPath + 'prediction/'
    prob_path = resultPath + 'reliability/'

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
    load_model_name = '../Model/tiramisu_fc_dense_' + model_name + '.json'
    model_file = open(load_model_name, 'r')
    tiramisu = models.model_from_json(model_file.read())
    model_file.close()
    
    # load final weight
    tiramisu.load_weights(result_path + '/prop_tiramisu_weights_' + model_name + '.best.hdf5')

    # -------------------------------------------------------------------------------------------- #
    conf_mat = np.zeros((5, 5))
    im_name_list = hdf5_file["im_name"].value
    test_set_list = sorted(test_set_list)

    for i_period in range(len(test_set_list)):
        for i_file in test_set_list[i_period]:  # the name of the image the cropped patch belongs to
            print(i_file)
            im_0_hdf5 = np.where(im_name_list==i_file)[0][0]

            im = hdf5_file["im"][im_0_hdf5]
            annot_img = hdf5_file["label"][im_0_hdf5]

            conf_temp = np.zeros((5, 5))
            pred_sum = np.zeros((dim1_, dim2_))

            pred_im = np.zeros((dim1_, dim2_, 4))

            for i_crop in range(test_crop_per_im):
                r_start, c_start = get_crop(dim_patch, dim_patch, dim2_, overlap, i_crop)
                patch = im[r_start:r_start + dim_patch, c_start:c_start + dim_patch, :]
                label_patch = annot_img[r_start:r_start + dim_patch, c_start:c_start + dim_patch]

                if 0 in label_patch:
                    pred_sum[r_start:r_start + dim_patch, c_start:c_start + dim_patch] = np.ones((dim_patch, dim_patch))
                else:
                    X = patch - test_mean

                    # predict on batch (batch size 1)
                    X = np.expand_dims(X, 0)
                    pred_label = tiramisu.predict_on_batch(X)

                    pred_im[r_start:r_start + dim_patch, c_start:c_start + dim_patch, :] += pred_label[0, :, :, :]
                    pred_sum[r_start:r_start + dim_patch, c_start:c_start + dim_patch] += np.sum(pred_label[0, :, :, :],
                                                                                                 axis=-1)
            label_image = pred_im / np.expand_dims(pred_sum, axis=-1)
            label_map = one_hot_reverse(label_image, color_label)

            cv2.imwrite(pred_path + i_file + '.png', label_map)
            prob_image = np.amax(label_image, axis=-1) * 255
            cv2.imwrite(prob_path + i_file + '.png', prob_image)

            # evaluation (confusion matrix)
            id_pred = colorToclass(label_map[:, :, 0])
            id_test = colorToclass(annot_img)

            for r in range(dim1_):
                for c in range(dim2_):
                    conf_temp[int(id_test[r, c]), int(id_pred[r, c])] += 1
            conf_mat += conf_temp

    print(conf_mat)
    np.save(resultPath + 'conf_mat', conf_mat)

    # -------------------------------------------------------------------------------------------------------------------#
    # precision, recall, overall accuracy, IOU
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

    # ---------------------------------------------------------------------------------------------------------------------#
    # record the results in the log file
    text_file = open(resultPath+'result.txt',"w")

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


def main():
    args = parser.parse_args()

    prediction(result_path=args.result_dir, model_name=args.model_name, pred_file=args.hdf5_file, dim_patch=args.dim_patch)

    
if __name__=='__main__':
    main()

