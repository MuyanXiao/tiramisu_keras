from __future__ import absolute_import
from __future__ import print_function

import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import MaxPooling2D, UpSampling2D, Cropping2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Conv2DTranspose, Conv3D, Lambda
from keras import backend as K

from keras.models import Model
from keras.layers import Input, concatenate, Add
from keras.regularizers import l2
import json

K.set_image_dim_ordering('tf')

class Tiramisu():

    def __init__(self):
        self.create([224,224], [4, 7, 12], 15, [12, 7, 4], 16, 0.0001, 0.5, 5, '69')

    def dense_block(self, layers_count, filters, input_layer, model_layers, l2_reg, dropout_rate, level):
        model_layers[level] = {}
        for i in range(layers_count):
            model_layers[level]['b_norm'+str(i+1)] = BatchNormalization(
                axis=-1,
                gamma_regularizer=l2(l2_reg),
                beta_regularizer=l2(l2_reg))(input_layer)
            model_layers[level]['act'+str(i+1)] = Activation('relu')(model_layers[level]['b_norm'+str(i+1)])
            model_layers[level]['conv'+str(i+1)] = Conv2D(
                filters,   kernel_size=(3, 3), padding='same',
                kernel_initializer="he_uniform",
                data_format='channels_last')(model_layers[level]['act'+str(i+1)])\

            model_layers[level]['drop_out'+str(i+1)] = Dropout(dropout_rate)(model_layers[level]['conv'+str(i+1)])

            input_layer = concatenate([input_layer, model_layers[level]['drop_out'+str(i+1)]], axis = -1)
            if i == 0:
                output_layer = model_layers[level]['drop_out'+str(i+1)]
            else:
                output_layer = concatenate([output_layer, model_layers[level]['drop_out'+str(i+1)]],axis=-1)

        return output_layer 

    def transition_down(self, filters, previous_layer, model_layers, l2_reg, dropout_rate, level):
        model_layers[level] = {}
        model_layers[level]['b_norm'] = BatchNormalization(
            axis=-1,
            gamma_regularizer=l2(l2_reg),
            beta_regularizer=l2(l2_reg))(previous_layer)
        model_layers[level]['act'] = Activation('relu')(model_layers[level]['b_norm'])
        model_layers[level]['conv'] = Conv2D(
            filters, kernel_size=(1, 1),
            padding='same',
            kernel_initializer="he_uniform")(model_layers[level]['act'])  # kernel size changed from 1 to 3

        print(model_layers[level]['conv'].name)

        model_layers[level]['drop_out'] = Dropout(dropout_rate)(model_layers[level]['conv'])
        model_layers[level]['avg_pool'] = AveragePooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            data_format='channels_last')(model_layers[level]['drop_out'])

        print(model_layers[level]['avg_pool'].shape)

        return model_layers[level]['avg_pool']

    def transition_up(self,filters,previous_layer, model_layers, level):
        model_layers[level] = {}
        model_layers[level]['conv'] = Conv2DTranspose(
            filters,  kernel_size=(1, 1), strides=(2, 2),
            padding='same',
            kernel_initializer="he_uniform",
            data_format='channels_last')(previous_layer)
        print(model_layers[level]['conv'].shape)

        return model_layers[level]['conv']

    def create(self, input_dim, layer_down, bottle_neck_layer, layer_up, growth_rate, l2_reg, dropout_rate, compression_rate, model_name):
        inputs = Input(([0], input_dim[1], 3))

        # first
        first_conv = Conv2D(
            48, kernel_size=5, padding='same',
            kernel_initializer="he_uniform",
            kernel_regularizer=l2(l2_reg),
            data_format='channels_last')(inputs)

        enc_model_layers = {}
        skip_connection = []

        # transition down
        input_layer = first_conv
        for i in range(len(layer_down)):
            layer_db = self.dense_block(layer_down[i], growth_rate, input_layer, enc_model_layers ,l2_reg, dropout_rate, 'db_'+str(i)+'_down')
            layer_concat = concatenate([input_layer,layer_db],axis=-1)
            layer_td = self.transition_down(int(layer_concat.shape[-1].value/compression_rate),layer_concat, enc_model_layers, l2_reg, dropout_rate,'TD_'+str(i))
            input_layer = layer_td

            skip_connection.append(layer_concat)

        skip_connection = skip_connection[::-1]

        # bottle neck
        layer_bottleneck = self.dense_block(bottle_neck_layer, growth_rate, input_layer ,enc_model_layers, l2_reg, dropout_rate, 'layer_bottleneck')
        print(layer_bottleneck.shape)

        tu_block = layer_bottleneck

        # transition up
        layer_up.append(bottle_neck_layer)
        for i in range(len(layer_up)-1):
            nb_tu_layer = layer_up[i-1]*growth_rate

            layer_tu = self.transition_up(int(nb_tu_layer/compression_rate), tu_block, enc_model_layers, 'TU_'+str(i))
            layer_concat_skip = concatenate([layer_tu, skip_connection[i]], axis=-1)
            layer_db = self.dense_block(layer_up[i], growth_rate, layer_concat_skip, enc_model_layers, l2_reg, dropout_rate, 'db_'+str(i)+'_up')
            tu_block = layer_db

        layer_up_last = concatenate([layer_concat_skip, layer_db], axis=-1, name='layer_up_last')

        # last
        last_conv_lake = Conv2D(4, activation='linear',
                                kernel_size=3,
                                padding='same',
                                kernel_regularizer = l2(l2_reg),
                                data_format='channels_last',
                                name='lake_lastconv4')(layer_up_last)
        act_lake = Activation('softmax',name='lake_softmax')(last_conv_lake)

        model = Model(inputs=[inputs], outputs=[act_lake])

        with open('../Model/tiramisu_fc_dense'+model_name+'.json', 'w') as outfile:
            outfile.write(json.dumps(json.loads(model.to_json()), indent=3))

        text_file = open('../Model/Setting.txt',"w")
        text_file.write("Input dimension: %d\n" % input_dim[0])
        text_file.write("DenseBlock Layers: ")
        for db_layer_log in layer_down:
           text_file.write("%d, "% db_layer_log)
        text_file.write("\nBottle Neck Layer: %d\n" % bottle_neck_layer)
        text_file.write("Growth rate: %d\n" % growth_rate)
        text_file.write("L2_regularizor: %f\n" % l2_reg)
        text_file.write("Dropout Rate: %f\n" % dropout_rate)
        text_file.write("Compression Rate: 1/%f" % compression_rate)
        text_file.close()

# Tiramisu()
