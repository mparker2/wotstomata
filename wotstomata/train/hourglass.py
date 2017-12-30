from math import log as logn
import numpy as np

from keras import models, layers, optimizers, losses


def conv_batch(prev, num_channels, kernel_size, name, conv_type=None):
    if conv_type is None:
        conv_type = layers.Conv2D
    prev = conv_type(num_channels,
                     kernel_size=kernel_size,
                     name=name + '_conv',
                     padding='same')(prev)
    prev = layers.BatchNormalization(name=name + '_batch_norm')(prev)
    return prev


def conv_batch_relu(prev, num_channels, kernel_size, name, conv_type=None):
    if conv_type is None:
        conv_type = layers.Conv2D
    prev = conv_batch(prev, num_channels, kernel_size, name, conv_type)
    prev = layers.Activation('relu', name=name + '_relu')(prev)
    return prev


def residual_block(prev, num_channels, name, conv_type=None, dim=2):
    shortcut = prev
    prev = conv_batch_relu(prev,
                           num_channels,
                           kernel_size=(3, ) * dim,
                           name=name + '_1',
                           conv_type=conv_type)
    prev = conv_batch(prev,
                      num_channels,
                      kernel_size=(3, ) * dim,
                      name=name + '_2',
                      conv_type=conv_type)

    shortcut = conv_batch(shortcut,
                          num_channels,
                          kernel_size=(1, ) * dim,
                          name=name + '_shortcut',
                          conv_type=conv_type)

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


def initial_res_blocks(input_layer, input_shape, max_hg_shape,
                       num_conv_channels, maxpool_size):
    if len(input_shape) == 3:
        conv_type = layers.Conv3D
        dim = 3
        maxpool_type = layers.MaxPool3D
        _maxpool_size = (1, maxpool_size, maxpool_size)
        input_shape = input_shape[1:]
        reduce_3d = True
    else:
        conv_type = layers.Conv2D
        dim = 2
        maxpool_type = layers.MaxPool2D
        _maxpool_size = (maxpool_size, maxpool_size)
        reduce_3d = False

    prev = input_layer
    n_initial_layers = get_n_resamplings(
        input_shape, max_hg_shape, maxpool_size)
    for i in range(n_initial_layers):
        res_block_name = 'init_res_block_{}'.format(i + 1)
        prev = residual_block(prev,
                              num_conv_channels,
                              res_block_name,
                              conv_type=conv_type,
                              dim=dim)
        prev = maxpool_type(pool_size=_maxpool_size,
                            name=res_block_name + '_max_pool')(prev)
    if reduce_3d:
        prev = layers.MaxPool3D((3, 1, 1),
                                name=res_block_name + '_reduce_to_2d')(prev)
        prev = layers.Reshape(max_hg_shape + (num_conv_channels, ))(prev)
    return prev


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
    if n_layers:
        n_channels = channel_downsample_size ** n_layers
        for i in range(0, n_layers):
            prev = conv_batch_relu(
                prev,
                n_channels,
                kernel_size=(3, 3),
                name=name + '_{}_{}'.format(conv_layer_type, i),
                conv_type=Conv
            )
            prev = layers.UpSampling2D(
                upsample_size, name=name + '_upsampling_{}'.format(i))(prev)
            n_channels //= channel_downsample_size
    else:
        prev = conv_batch_relu(
           prev,
           num_channels=channel_downsample_size,
           kernel_size=(3, 3),
           name=name + '_{}'.format(conv_layer_type),
           conv_type=Conv
        )
        n_channels = 1
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


def feature_count_output(prev, num_convs, num_channels,
                         num_dense_units, final_activation, name):
    for i in range(num_convs):
        prev = conv_batch_relu(prev,
                               num_channels,
                               kernel_size=(3, 3),
                               name=name + '_{}'.format(i))
        prev = layers.MaxPool2D(2, name=name + '_{}_max_pool'.format(i))(prev)
    prev = layers.Flatten(name=name + '_flatten')(prev)
    prev = layers.Dense(num_dense_units, name=name + '_dense')(prev)
    prev = layers.Dense(1, name=name, activation=final_activation)(prev)
    return prev


def build_hourglass(num_hg_modules=4,
                    num_conv_channels=16,
                    input_shape=(256, 256),
                    color_mode='rgb',
                    max_hg_shape=(64, 64),
                    min_hg_shape=(4, 4),
                    resampling_size=2,
                    output_shape=(64, 64),
                    output_channel_downsampling_size=4,
                    transpose_output=False,
                    transpose_output_shape=(128, 128),
                    learning_rate=2.5e-4):
    if color_mode == 'rgb':
        input_channels = 3
    elif color_mode == 'grayscale':
        input_channels = 1
    else:
        raise TypeError('color_mode {} not recognised'.format(input_channels))
    input_layer = layers.Input(
        shape=input_shape + (input_channels,), name='input')
    outputs = []
    output_names = []

    # initial residual blocks to reduce size from input_shape to max_hg_shape
    prev = initial_res_blocks(
        input_layer=input_layer,
        input_shape=input_shape,
        max_hg_shape=max_hg_shape,
        num_conv_channels=num_conv_channels,
        maxpool_size=resampling_size)

    for i in range(num_hg_modules):
        module_name = 'hg_{}'.format(i)
        prev = hourglass_module(prev, num_conv_channels,
                                module_name, min_hg_shape)
        # create intermediate output from hourglass
        hm_o_name = 'hm_{}'.format(i + 1)
        hm_output = conv_outputs(
            prev=prev,
            input_shape=max_hg_shape,
            output_shape=output_shape,
            upsample_size=resampling_size,
            channel_downsample_size=output_channel_downsampling_size,
            name=hm_o_name,
            conv_layer_type='conv')
        outputs.append(hm_output)
        output_names.append(hm_o_name)

        if transpose_output:
            seg_o_name = 'seg_{}'.format(i + 1)
            seg_output = conv_outputs(
                prev=prev,
                input_shape=max_hg_shape,
                output_shape=transpose_output_shape,
                upsample_size=resampling_size,
                channel_downsample_size=output_channel_downsampling_size,
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


def transfer_learn_counts(hourglass,
                          num_hg_modules,
                          num_conv_channels,
                          transpose_output,
                          min_hg_shape,
                          max_hg_shape,
                          resampling_size,
                          learning_rate):

    res_block_n = get_n_resamplings(max_hg_shape,
                                    min_hg_shape,
                                    resampling_size)
    res_block_n = res_block_n * 2 + 1
    layer_name = 'hg_{:d}_res_block_{:d}_add'.format(num_hg_modules - 1,
                                                     res_block_n)
    prev = hourglass.get_layer(name=layer_name).output

    outputs = []

    hm_count_output = feature_count_output(
        prev,
        num_convs=2,
        num_channels=num_conv_channels,
        num_dense_units=8,
        final_activation=None,
        name='hm_c')
    outputs.append(hm_count_output)
    if transpose_output:
        seg_count_output = feature_count_output(
            prev,
            num_convs=2,
            num_channels=num_conv_channels,
            num_dense_units=8,
            final_activation=None,
            name='seg_c')
        outputs.append(seg_count_output)

        mask_output = feature_count_output(
            prev,
            num_convs=2,
            num_channels=num_conv_channels,
            num_dense_units=8,
            final_activation='sigmoid',
            name='blur')
        outputs.append(mask_output)

        loss = [losses.mean_squared_error,
                losses.mean_squared_error,
                losses.binary_crossentropy]
    else:
        loss = losses.mean_squared_error

    hourglass_t = models.Model(inputs=hourglass.input, outputs=outputs)

    for layer in hourglass_t.layers:
        if layer in hourglass.layers:
            layer.trainable = False

    hourglass_t.compile(
        optimizer=optimizers.RMSprop(lr=learning_rate),
        loss=loss,
    )

    return hourglass_t


def combine_models(original, transferred, learning_rate):
    assert original.input == transferred.input
    combined = models.Model(
        inputs=original.input,
        outputs=original.outputs + transferred.outputs
    )

    for layer in combined.layers:
        layer.trainable = True

    combined.compile(
        optimizer=optimizers.RMSprop(lr=learning_rate),
        loss=losses.mean_squared_error
    )

    return combined


def load_model(model_json_fn, model_weights_fn):
    with open(model_json_fn) as j:
        model = models.model_from_json(j.read())
    model.load_weights(model_weights_fn)
    return model


def summary_to_file(model, log_fn):
    with open(log_fn, 'w') as f:
        def log_line(line):
            f.write(line + '\n')
        model.summary(line_length=200, print_fn=log_line)
