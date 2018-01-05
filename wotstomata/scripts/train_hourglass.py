import sys
import os
import shutil
import pickle as pk
import yaml
import click

if sys.argv[1] == 'run':
    from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

    from wotstomata.train.hourglass import (
        build_hourglass,
        transfer_learn_counts,
        combine_models,
        load_model,
        summary_to_file
    )

    from wotstomata.train.img_generator import (
        generate_training_data,
        batch_formatting_wrapper,
        cell_counting_wrapper
    )

    from wotstomata.train.plotting import PlotLosses

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
    transfer_params = config['cell_counting_transfer']
    output_params = config['outputs']

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

    if (not transfer_params['load_model_for_transfer'] or
            not transfer_params['generate_feature_counts']):
        hg = build_hourglass(
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
            learning_rate=training_params['learning_rate']
        )
        summary_to_file(hg, output_params['summary_fn'])

        model_json = hg.to_json()
        with open(output_params['architechture_fn'], "w") as j:
            j.write(model_json)

        train_gen_f = batch_formatting_wrapper(
            train_gen,
            n_outputs=arch_params['num_hg_modules'],
            heatmap_shape=arch_params['heatmap_output_shape'],
            segment=arch_params['generate_segmented_cells'],
            segment_shape=arch_params['segment_output_shape']
        )

        val_gen_f = batch_formatting_wrapper(
            val_gen,
            n_outputs=arch_params['num_hg_modules'],
            heatmap_shape=arch_params['heatmap_output_shape'],
            segment=arch_params['generate_segmented_cells'],
            segment_shape=arch_params['segment_output_shape']
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

        history = hg.fit_generator(
            generator=train_gen_f,
            steps_per_epoch=training_params['steps_per_epoch'],
            validation_data=val_gen_f,
            validation_steps=training_params['validation_steps'],
            epochs=training_params['num_epochs'],
            verbose=training_params['verbose'],
            callbacks=[reduce_lr, plot_losses, checkpoint]
        )

        hg.save_weights(output_params['weights_fn'])
        with open(output_params['history_fn'], "wb") as h:
            pk.dump(history.history, h)

    else:
        hg = load_model(output_params['architechture_fn'],
                        output_params['weights_fn'])

    if transfer_params['generate_feature_counts']:
        hg_t = transfer_learn_counts(
            hg,
            num_hg_modules=arch_params['num_hg_modules'],
            num_conv_channels=arch_params['num_conv_channels'],
            transpose_output=arch_params['generate_segmented_cells'],
            min_hg_shape=arch_params['min_hg_shape'],
            max_hg_shape=arch_params['max_hg_shape'],
            resampling_size=arch_params['resampling_size'],
            learning_rate=training_params['learning_rate'],
        )
        train_gen_c = cell_counting_wrapper(train_gen)
        val_gen_c = cell_counting_wrapper(val_gen)

        checkpoint_t = ModelCheckpoint(
            filepath=output_params['transfer_checkpoint_fn'],
            monitor='val_loss',
            verbose=training_params['verbose'],
            save_best_only=True,
            mode='min',
            save_weights_only=True,
            period=10
        )

        history_t = hg_t.fit_generator(
            generator=train_gen_c,
            steps_per_epoch=training_params['steps_per_epoch'],
            validation_data=val_gen_c,
            validation_steps=training_params['validation_steps'],
            epochs=training_params['num_epochs'],
            verbose=training_params['verbose'],
            callbacks=[checkpoint_t, ]
        )

        hg_c = combine_models(hg, hg_t, training_params['learning_rate'])

        model_json = hg_c.to_json()
        with open(output_params['combined_architechture_fn'], "w") as j:
            j.write(model_json)

        hg_c.save_weights(output_params['combined_weights_fn'])

        with open(output_params['transfer_history_fn'], "wb") as h:
            pk.dump(history_t.history, h)
