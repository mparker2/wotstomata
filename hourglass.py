from math import log as logn
import numpy as np

from keras import models, layers, optimizers, losses


def residual_block(prev, num_channels, name):
    shortcut = prev
    prev = layers.Conv2D(num_channels,
                         kernel_size=(3, 3),
                         name=name + '_conv_1',
                         padding='same')(prev)
    prev = layers.BatchNormalization(name=name + '_batch_norm_1')(prev)
    prev = layers.Activation('relu', name=name + '_relu')(prev)

    prev = layers.Conv2D(num_channels,
                         kernel_size=(3, 3),
                         name=name + '_conv_2',
                         padding='same')(prev)
    prev = layers.BatchNormalization(name=name + '_batch_norm_2')(prev)

    shortcut = layers.Conv2D(num_channels,
                             kernel_size=(1, 1),
                             name=name + '_shortcut_conv',
                             padding='same')(shortcut)
    shortcut = layers.BatchNormalization(
        name=name + '_shortcut_batch_norm')(shortcut)

    prev = layers.add([shortcut, prev], name=name + '_add_shortcut')
    prev = layers.Activation('relu', name=name + '_final_relu')(prev)
    return prev


def get_n_resamplings(input_shape, output_shape, upsample_size):
    n = []
    for i, o in zip(input_shape, output_shape):
        n.append(abs(logn(o / i, upsample_size)))
    assert all([x == n[0] for x in n])
    assert n[0] == int(n[0])
    return int(n[0])


def conv_outputs(prev, input_shape, output_shape,
                 upsample_size, channel_downsample_size,
                 name, conv_layer_type='conv'):
    if conv_layer_type == 'conv':
        Conv = layers.Conv2D
    elif conv_layer_type == 'convtranspose':
        Conv = layers.Conv2DTranspose
    else:
        raise TypeError(
            'conv_layer_type {} not recognised'.format(conv_layer_type))
    n_layers = get_n_resamplings(input_shape, output_shape, upsample_size)
    n_channels = channel_downsample_size ** n_layers
    for i in range(0, n_layers):
        prev = Conv(n_channels, kernel_size=3, padding='same',
                    name=name + '_{}_{}'.format(conv_layer_type, i))(prev)
        prev = layers.BatchNormalization(
            name=name + '_batch_norm_{}'.format(i))(prev)
        prev = layers.Activation('relu',
                                 name=name + '_activation_{}'.format(i))(prev)
        prev = layers.UpSampling2D(
            upsample_size, name=name + '_upsampling_{}'.format(i))(prev)
        n_channels //= channel_downsample_size
    assert n_channels == 1
    prev = Conv(n_channels, kernel_size=1, padding='same', name=name)(prev)
    return prev


def hourglass_module(prev, num_channels, module_name, min_shape=(4, 4)):
    convs = []
    i = 1
    while True:
        res_block_name = '{}_res_block_{}'.format(module_name, i)
        prev = residual_block(prev, num_channels, res_block_name)
        curr_shape = tuple(prev.shape.as_list()[1:-1])
        if curr_shape == min_shape:
            break
        convs.append(prev)
        prev = layers.MaxPool2D(2, name=res_block_name + '_max_pool')(prev)
        i += 1

    i += 1
    for conv in convs[::-1]:
        res_block_name = '{}_res_block_{}'.format(module_name, i)
        conv = residual_block(conv, num_channels,
                              res_block_name + '_hg_shortcut')
        prev = residual_block(prev, num_channels, res_block_name)
        prev = layers.UpSampling2D(2, name=res_block_name + '_upsampled')(prev)
        prev = layers.Add(name=res_block_name + '_add')([conv, prev])
        i += 1

    return prev


def build_hourglass(num_hg_modules=4,
                    num_conv_channels=16,
                    input_shape=(256, 256),
                    color_mode='rgb',
                    max_hg_shape=(64, 64),
                    min_hg_shape=(4, 4),
                    output_shape=(64, 64),
                    transpose_output=False,
                    transpose_output_shape=(128, 128),
                    learning_rate=2.5e-4):
    if color_mode == 'rgb':
        input_channels = 3
    elif color_mode == 'grayscale':
        input_channels = 1
    else:
        raise TypeError('color_mode {} not recognised'.format(input_channels))
    input_layer = layers.Input(shape=input_shape + (input_channels,))
    outputs = []
    output_names = []
    prev = input_layer
    # initial residual blocks to reduce size from input_shape to max_hg_shape
    n_initial_layers = get_n_resamplings(input_shape, max_hg_shape, 2)
    for i in range(n_initial_layers):
        res_block_name = 'init_res_block_{}'.format(i + 1)
        prev = residual_block(prev, num_conv_channels, res_block_name)
        prev = layers.MaxPool2D(2, name=res_block_name + '_max_pool')(prev)

    for i in range(num_hg_modules):
        module_name = 'hg_{}'.format(i)
        prev = hourglass_module(prev, num_conv_channels,
                                module_name, min_hg_shape)

        # create intermediate output from hourglass
        hm_o_name = 'hm_{}'.format(i + 1)
        hm_output = conv_outputs(
            y=prev,
            input_shape=max_hg_shape,
            output_shape=output_shape,
            upsample_size=2,
            channel_downsample_size=4,
            name=hm_o_name,
            conv_layer_type='conv')
        outputs.append(hm_output)
        output_names.append(hm_o_name)

        if transpose_output:
            seg_o_name = 'seg_{}'.format(i + 1)
            seg_output = conv_outputs(
                y=prev,
                input_shape=max_hg_shape,
                output_shape=transpose_output_shape,
                upsample_size=2,
                channel_downsample_size=4,
                name=seg_o_name,
                conv_layer_type='convtranspose')
            outputs.append(seg_output)
            output_names.append(seg_o_name)

    loss_weights = np.cumsum(np.ones(num_hg_modules))
    if transpose_output:
        loss_weights = np.repeat(loss_weights, 2)
    loss_weights /= loss_weights.sum()
    loss_weights = {layer: w for layer, w in zip(output_names, loss_weights)}
    hourglass = models.Model(inputs=input_layer, outputs=outputs)
    hourglass.compile(
        optimizer=optimizers.RMSprop(lr=learning_rate),
        loss=losses.mean_squared_error,
        loss_weights=loss_weights)
    return hourglass


def load_model(model_json_fn, model_weights_fn):
    with open(model_json_fn) as j:
        model = models.model_from_json(j.read())
    model.load_weights(model_weights_fn)
    return model
