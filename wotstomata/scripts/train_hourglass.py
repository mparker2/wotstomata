import sys
import os
import shutil
import pickle as pk
from datetime import datetime
import yaml
import click

if sys.argv[1] == 'run':
    import tensorflow as tf
    from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

    from ..train.hourglass import (
        build_hourglass,
        create_multi_gpu_model,
        get_available_gpus,
        compile_hourglass,
        summary_to_file
    )

    from ..train.img_generator import (
        generate_training_data,
        batch_formatting_wrapper,
    )

    from ..train.plotting import PlotLosses

elif sys.argv[1] == 'config':
    import wotstomata


@click.group()
def main():
    pass


@main.command()
@click.argument('new-config-fn', required=False, default='./train.yaml')
def config(new_config_fn):
    directory = os.path.split(wotstomata.__file__)[0]
    config_fn = os.path.join(directory, 'data', 'train.yaml')
    shutil.copy(config_fn, new_config_fn)


@main.command()
@click.argument('config-fn', required=False, default='./train.yaml')
def run(config_fn):
    with open(config_fn) as c:
        config = yaml.load(c)
    arch_params = config['model_architechture']
    training_params = config['model_training']
    training_data = config['training_data']
    output_params = config['outputs']

    now = datetime.now().replace(microsecond=0)
    for k, v in output_params.items():
        if '{date}' in v:
            output_params[k] = v.format(
                date='{:04d}-{:02d}-{:02d}'.format(*now.isocalendar()))
        elif '{datetime}' in v:
            output_params[k] = v.format(datetime=now.isoformat())

    train_gen = generate_training_data(
        directory=training_data['training_data_dir'],
        batch_size=training_params['minibatch_size'],
        segment=arch_params['generate_segmented_cells'],
        frac_randomly_sampled_imgs=training_params[
            'frac_randomly_sampled_imgs'],
        input_3d=len(arch_params['input_shape']) == 3
    )

    val_gen = generate_training_data(
        directory=training_data['val_data_dir'],
        batch_size=training_params['minibatch_size'],
        segment=arch_params['generate_segmented_cells'],
        frac_randomly_sampled_imgs=training_params[
            'frac_randomly_sampled_imgs'],
        input_3d=len(arch_params['input_shape']) == 3
    )

    gpus = get_available_gpus()
    # if no gpus, build on cpu, if multi-gpu, build on cpu then create
    # multi-gpu copy of model.
    device = '/cpu:0' if training_params['num_gpus'] != 1 else gpus[0]
    with tf.device(device):
        hg_cpu, output_names = build_hourglass(
            num_hg_modules=arch_params['num_hg_modules'],
            num_conv_channels=arch_params['num_conv_channels'],
            input_shape=arch_params['input_shape'],
            max_hg_shape=arch_params['max_hg_shape'],
            min_hg_shape=arch_params['min_hg_shape'],
            resampling_size=arch_params['resampling_size'],
            output_shape=arch_params['heatmap_output_shape'],
            output_channel_downsampling_size=arch_params[
                'output_channel_downsampling_size'],
            transpose_output=arch_params['generate_segmented_cells'],
            transpose_output_shape=arch_params['segment_output_shape'],
            blur_output=arch_params['estimate_blur_output']
        )
    hg_gpu = create_multi_gpu_model(hg_cpu, training_params['num_gpus'])
    compile_hourglass(
        hg_gpu,
        output_names,
        num_hg_modules=arch_params['num_hg_modules'],
        transpose_output=arch_params['generate_segmented_cells'],
        blur_output=arch_params['estimate_blur_output'],
        learning_rate=training_params['learning_rate']
    )
    summary_to_file(hg_cpu, output_params['summary_fn'])

    model_json = hg_cpu.to_json()
    with open(output_params['architechture_fn'], "w") as j:
        j.write(model_json)

    train_gen_f = batch_formatting_wrapper(
        train_gen,
        n_outputs=arch_params['num_hg_modules'],
        heatmap_shape=arch_params['heatmap_output_shape'],
        segment=arch_params['generate_segmented_cells'],
        segment_shape=arch_params['segment_output_shape'],
        in_focus=arch_params['estimate_blur_output']
    )

    val_gen_f = batch_formatting_wrapper(
        val_gen,
        n_outputs=arch_params['num_hg_modules'],
        heatmap_shape=arch_params['heatmap_output_shape'],
        segment=arch_params['generate_segmented_cells'],
        segment_shape=arch_params['segment_output_shape'],
        in_focus=arch_params['estimate_blur_output']
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=20,
        cooldown=20,
        verbose=training_params['verbose'],
        epsilon=0.0001
    )

    plot_val_data, _ = next(val_gen_f)
    plot_losses = PlotLosses(
        output_params['plot_fn'],
        plot_val_data,
        training_params['steps_per_epoch'],
        input_3d=len(arch_params['input_shape']) == 3,
        layer=1
    )

    checkpoint = ModelCheckpoint(
        filepath=output_params['checkpoint_fn'],
        monitor='val_loss',
        verbose=training_params['verbose'],
        save_best_only=True,
        mode='min',
        save_weights_only=True,
        period=10
    )

    history = hg_gpu.fit_generator(
        generator=train_gen_f,
        steps_per_epoch=training_params['steps_per_epoch'],
        validation_data=val_gen_f,
        validation_steps=training_params['validation_steps'],
        epochs=training_params['num_epochs'],
        verbose=training_params['verbose'],
        callbacks=[reduce_lr, plot_losses, checkpoint]
    )

    hg_cpu.save_weights(output_params['weights_fn'])
    with open(output_params['history_fn'], "wb") as h:
        pk.dump(history.history, h)
