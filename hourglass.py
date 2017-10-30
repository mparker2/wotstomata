import numpy as np

import keras
from keras import models, layers, optimizers, losses


def residual_block(y, num_channels, name):
    shortcut = y
    y = layers.Conv2D(num_channels,
                      kernel_size=(3, 3),
                      name=name + '_conv_1',
                      padding='same')(y)
    y = layers.BatchNormalization(name=name + '_batch_norm_1')(y)
    y = layers.Activation('relu', name=name + '_relu')(y)

    y = layers.Conv2D(num_channels,
                      kernel_size=(3, 3),
                      name=name + '_conv_2',
                      padding='same')(y)
    y = layers.BatchNormalization(name=name + '_batch_norm_2')(y)

    shortcut = layers.Conv2D(num_channels,
                             kernel_size=(1, 1),
                             name=name + '_shortcut_conv',
                             padding='same')(shortcut)
    shortcut = layers.BatchNormalization(
        name=name + '_shortcut_batch_norm')(shortcut)

    y = layers.add([shortcut, y], name=name + '_add_shortcut')
    y = layers.Activation('relu', name=name + '_final_relu')(y)

    return y


def output_convs(y, num_conv_channels, name):
    name = 'output_{}'.format(name)
    y = layers.Conv2D(num_conv_channels,
                      kernel_size=1,
                      padding='same',
                      name=name + '_conv_1')(y)
    y = layers.BatchNormalization(name=name + '_batch_norm')(y)
    y = layers.Activation('relu', name=name + '_activation')(y)
    y = layers.Conv2D(1, kernel_size=1,
                      padding='same',
                      name=name)(y)
    return y, name


def hourglass_module(y, num_channels,
                     module_name, min_shape=4):
    prev = y
    convs = []
    i = 1
    while True:
        res_block_name = '{}_res_block_{}'.format(module_name, i)
        prev = residual_block(prev, num_channels, res_block_name)
        _, *curr_shape, _ = keras.backend.int_shape(prev)
        if curr_shape[0] == min_shape or curr_shape[1] == min_shape:
            break
        convs.append(prev)
        prev = layers.MaxPool2D(2, name=res_block_name + '_max_pool')(prev)
        i += 1

    i += 1
    for conv in convs[::-1]:
        res_block_name = '{}_res_block_{}'.format(module_name, i)
        prev = residual_block(prev, num_channels, res_block_name)
        prev = layers.UpSampling2D(2, name=res_block_name + '_upsampled')(prev)
        prev = layers.Add(name=res_block_name + '_add')([conv, prev])
        i += 1

    return prev


def build_hourglass(num_hg_modules=4,
                    num_conv_channels=16,
                    min_shape=4,
                    hg_dropout=0.4):
    input_layer = layers.Input(shape=(256, 256, 3))
    outputs = []
    output_names = []
    prev = input_layer
    # initial residual blocks to reduce img size from 256 to 64
    for i in range(2):
        res_block_name = 'init_res_block_{}'.format(i + 1)
        prev = residual_block(prev, num_conv_channels, res_block_name)
        prev = layers.MaxPool2D(2, name=res_block_name + '_max_pool')(prev)

    for i in range(num_hg_modules):
        module_name = 'hg_{}'.format(i)
        prev = hourglass_module(prev, num_conv_channels,
                                module_name, min_shape)
        
        # create intermediate output from hourglass
        output, o_name = output_convs(prev, num_conv_channels, i + 1)
        outputs.append(output)
        output_names.append(o_name)

    loss_weights = np.cumsum(np.ones(num_hg_modules))
    loss_weights /= loss_weights.sum()
    loss_weights = {layer: w for layer, w in zip(output_names, loss_weights)}
    hourglass = models.Model(inputs=input_layer, outputs=outputs)
    hourglass.compile(optimizer=optimizers.RMSprop(lr=2.5e-4),
                      loss=losses.mean_squared_error,
                      loss_weights=loss_weights)
    return hourglass
