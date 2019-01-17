from __future__ import absolute_import
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import keras.models as models
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.callbacks import ModelCheckpoint, LambdaCallback, BaseLogger, EarlyStopping
# from keras.models import Model
from keras import backend as K
from keras import callbacks
from keras.callbacks import Callback
# import math
# import cv2
# import numpy as np
# import json
import h5py
# import random 

from helper import *

K.set_image_dim_ordering('tf')  # channel_last data format

# from data_loader import DataLoader
from dataGenerator import DataGenerator
from modelTiramisu import Tiramisu
import math


def basicTest(hdf5_name='C1_17_RS', rs_rate=4, balancing=False):
    # random.seed(7)
    # ------------------------------------------------------------------------------------------------------------------- #
    # Read HDF5 file
    hdf5_name = hdf5_name
    hdf5_file = h5py.File('../Data/'+hdf5_name+'.hdf5', 'r')

    # # ------------------------------------------------------------------------------------------------------------------ #
    # Generate training, validation, testing sets
    # the separation settings are meanwhile saved to the HDF5 file
    # data_set_list = DataLoader('../Data/C6/All/').generate()

    # ------------------------------------------------------------------------------------------------------------------- #
    # Generate batch sample
    # settings
    batch_size = 10
    dim_patch = 224
    num_fusion = 0  # the number of previous/following patch to be fused in the input

    params_train = {
        'hdf5_file': hdf5_file,
        'dim_x': dim_patch,
        'dim_y': dim_patch,
        'dim_z': 3,
        'batch_size': batch_size,
        'num_fusion': num_fusion,
        'tag': 'Train',
        'aug': rs_rate,
        'balancing': balancing
    }

    params_val = {
        'hdf5_file': hdf5_file,
        'dim_x': dim_patch,
        'dim_y': dim_patch,
        'dim_z': 3,
        'batch_size': batch_size,
        'num_fusion': num_fusion,
        'tag': 'Val'
    }

    train_setting = DataGenerator(**params_train)
    val_setting = DataGenerator(**params_val)

    # calculate class weighting
    # class_weighting = hdf5_file["class_weighting"].value
    class_weighting = train_setting.class_weight
    num_train = len(train_setting.crop_list)
    num_val = len(val_setting.crop_list)

    # Generators
    training_generator = train_setting.generate()
    validation_generator = val_setting.generate()

    # ------------------------------------------------------------------------------------------------------------------- #
    # record the settings,training process,results in a file
    import time
    curr_date = time.strftime("%d/%m/%Y")
    curr_time = time.strftime("%H:%M:%S")
    file_id = curr_date[0:2]+curr_date[3:5]+'_'+curr_time[0:2]

    # file_id = 'test'
    # create a new folder
    newPath = '../Result/'+file_id+'_C1_7S/'

    try:
        os.makedirs(newPath)
    except OSError:
        if not os.path.isdir(newPath):
            raise

    np.save(newPath + 'mean_train', train_setting.mean)

    text_file = open(newPath+'Log.txt', "w")
    text_file.write("Date: %s\n" % curr_date)
    text_file.write("Start time: %s\n\n" % curr_time)

    # ------------------------------------------------------------------------------------------------------------------- #
    # load the model (and weights):
    model_id = hdf5_name
    # Tiramisu_3D().create([dim_patch, dim_patch], [4, 6, 8], 10, [8, 6, 4], 12, 0.0001, 0.5, 5, model_id)
    Tiramisu().create([dim_patch, dim_patch], [4, 6, 8], 10, [8, 6, 4], 12, 0.0001, 0.5, 5, model_id)

    load_model_name = '../Model/tiramisu_fc_dense'+model_id+'.json'
    with open(load_model_name) as model_file:
        tiramisu = models.model_from_json(model_file.read())

    # tiramisu.load_weights("../Result/2805_11_C1_RS/prop_tiramisu_weights_C1_17_RS.best.hdf5",by_name=True)

    # specify optimizer
    optimizer = Nadam(lr=0.0005)

    nb_epoch = 20

    # metrics using accuracy or IoU
    tiramisu.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # checkpoint 278
    TensorBoard = callbacks.TensorBoard(log_dir='./logs', histogram_freq=5, write_graph=True, write_images=True)

    filePath = newPath+'prop_tiramisu_weights_'+model_id+'.best.hdf5'
    checkpoint = ModelCheckpoint(filePath, monitor='val_acc', verbose=2, save_best_only=True, save_weights_only=False, mode='max')
    earlyStopping = EarlyStopping(monitor='val_acc', patience=4, verbose=1, mode='max')

    callbacks_list = [checkpoint, earlyStopping]


    # class callbackLog(Callback):
    #     def __init__(self):
    #         super(callbackLog, self).__init__()
    #         self.seen = []
    #
    #     def on_batch_end(self, batch, logs=None):
    #         logs = logs or {}
    #         batch_size = logs.get('size', 0)
    #         self.seen.append(batch_size)
    #
    # batch_print_callback = callbackLog()

    # ------------------------------------------------------------------------------------------------------------------- #
    # record the settings in a text file
    with open('../Model/Setting.txt', 'r') as model_setting_log:
        model_setting = model_setting_log.read()

    text_file.write(model_setting)
    text_file.write("\nData set: C1_17\n")
    text_file.write('\nBalancing: %s' % str(train_setting.balancing))
    text_file.write('\nSampling rate: %d' % train_setting.aug)

    text_file.write("\nclass weights:\n")
    for w_item in class_weighting:
        text_file.write("%f\n" % w_item)

    text_file.write("\n# training data: %d\n" % num_train)
    text_file.write("# validation data: %d\n" % num_val)
    text_file.write("model: %s\n" % load_model_name)
    # text_file.write("loaded the weights: %s\n\n"%"1303_19/prop_tiramisu_weights_69.best.hdf5")

    text_file.write("optimizer = Nadam(lr=0.0005)\n")
    text_file.write("loss function: categorical_crossentropy\n\n")
    text_file.write("# epoch: %d\n batch size: %d\n" % (nb_epoch, batch_size))
    text_file.write("weights stored in: %s\n" % filePath)
    text_file.write("number of stacks: %d\n" % (num_fusion*2 + 1))
    text_file.write("no floating ice in training set")
    text_file.close()


    # ------------------------------------------------------------------------------------------------------------------- #
    # Fit the model
    history = tiramisu.fit_generator(
        generator=training_generator,  # gen(train_setting),
        steps_per_epoch=math.ceil(num_train/float(batch_size)),
        validation_data=validation_generator,  # gen(val_setting),
        validation_steps=math.ceil(num_val/float(batch_size)),
        epochs=nb_epoch,
        verbose=1,
        # max_queue_size=1,
        # use_multiprocessing=True,
        callbacks=callbacks_list)
        # class_weight=class_weighting)

    # print(batch_print_callback.seen)

    # This save the trained model weights to this file with number of epochs
    tiramisu.save_weights(newPath+'prop_tiramisu_weights_'+model_id+'_{}.hdf5'.format(nb_epoch))

    # ------------------------------------------------------------------------------------------------------------------- #
    # plot and save training history
    import matplotlib.pyplot as plt
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.figure(1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(newPath+'acc.png')
    # summarize history for loss
    plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(newPath+'loss.png')

    # save history to numpy array
    log_history = [[]]*4
    log_history[0] = np.array(history.history['loss'])
    log_history[1] = np.array(history.history['val_loss'])
    log_history[2] = np.array(history.history['acc'])
    log_history[3] = np.array(history.history['val_acc'])

    np.save(newPath+'Train_history', log_history)

    hdf5_file.close()


if __name__=='__main__':
    basicTest()