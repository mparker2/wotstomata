import pickle as pk

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from hourglass import (
    build_hourglass,
    transfer_learn_counts,
    combine_models,
    load_model,
    summary_to_file
)

from img_generator import (
    generate_training_data,
    batch_formatting_wrapper,
    cell_counting_wrapper
)

from plotting import PlotLosses


NUM_HG_MODULES = 4
NUM_CONV_CHANNELS = 16
INPUT_SHAPE = (3, 256, 256)
USE_3D_INPUT = len(INPUT_SHAPE) == 3
MAX_HG_SHAPE = (64, 64)
MIN_HG_SHAPE = (4, 4)
RESAMPLING_SIZE = 2
OUTPUT_CHANNEL_DOWNSAMPLING_SIZE = 4

GENERATE_SEGMENTED_CELLS = True
HEATMAP_OUTPUT_SHAPE = (64, 64)
SEGMENT_OUTPUT_SHAPE = (128, 128)

GENERATE_FEATURE_COUNTS = True
LOAD_MODEL_FOR_TRANSFER = True

LEARNING_RATE = 2.5e-4
MINIBATCH_SIZE = 32
STEPS_PER_EPOCH = 128
VALIDATION_STEPS = 12
NUM_EPOCHS = 75
NUM_TRANSFER_EPOCHS = 50

FRAC_RANDOMLY_SAMPLED_IMGS = 0.4
VERBOSE = 1

TRAINING_DATA_DIR = 'train'
VAL_DATA_DIR = 'val'

PLOT_FN = "./segmentation_loss.png"
ARCHITECTURE_FN = "segmentation_model_architecture.json"
SUMMARY_FN = "segmentation_model_summary.txt"
CHECKPOINT_FN = "segmentation_model_checkpoint_weights.h5"
WEIGHTS_FN = "segmentation_model_final_weights.h5"
HISTORY_FN = "segmentation_model_history.pk"

TRANSFER_CHECKPOINT_FN = 'count_model_checkpoint_weights.h5'
TRANSFER_HISTORY_FN = "count_model_history.pk"
COMBINED_ARCHITECTURE_FN = 'segmentation_count_model_architecture.json'
COMBINED_WEIGHTS_FN = 'segmentation_count_model_final_weights.h5'


if __name__ == '__main__':

    train_gen = generate_training_data(
        directory=TRAINING_DATA_DIR,
        batch_size=MINIBATCH_SIZE,
        segment=GENERATE_SEGMENTED_CELLS,
        frac_randomly_sampled_imgs=FRAC_RANDOMLY_SAMPLED_IMGS,
        input_3d=USE_3D_INPUT
    )

    val_gen = generate_training_data(
        directory=VAL_DATA_DIR,
        batch_size=MINIBATCH_SIZE,
        segment=GENERATE_SEGMENTED_CELLS,
        frac_randomly_sampled_imgs=FRAC_RANDOMLY_SAMPLED_IMGS,
        input_3d=USE_3D_INPUT
    )

    if not LOAD_MODEL_FOR_TRANSFER or not GENERATE_FEATURE_COUNTS:
        hg = build_hourglass(
            num_hg_modules=NUM_HG_MODULES,
            num_conv_channels=NUM_CONV_CHANNELS,
            input_shape=INPUT_SHAPE,
            max_hg_shape=MAX_HG_SHAPE,
            min_hg_shape=MIN_HG_SHAPE,
            resampling_size=RESAMPLING_SIZE,
            output_shape=HEATMAP_OUTPUT_SHAPE,
            output_channel_downsampling_size=OUTPUT_CHANNEL_DOWNSAMPLING_SIZE,
            transpose_output=GENERATE_SEGMENTED_CELLS,
            transpose_output_shape=SEGMENT_OUTPUT_SHAPE,
            learning_rate=LEARNING_RATE
        )
        summary_to_file(hg, SUMMARY_FN)

        model_json = hg.to_json()
        with open(ARCHITECTURE_FN, "w") as j:
            j.write(model_json)

        train_gen_f = batch_formatting_wrapper(
            train_gen,
            n_outputs=NUM_HG_MODULES,
            heatmap_shape=HEATMAP_OUTPUT_SHAPE,
            segment=GENERATE_SEGMENTED_CELLS,
            segment_shape=SEGMENT_OUTPUT_SHAPE
        )

        val_gen_f = batch_formatting_wrapper(
            val_gen,
            n_outputs=NUM_HG_MODULES,
            heatmap_shape=HEATMAP_OUTPUT_SHAPE,
            segment=GENERATE_SEGMENTED_CELLS,
            segment_shape=SEGMENT_OUTPUT_SHAPE
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=20,
            cooldown=20,
            verbose=VERBOSE,
            epsilon=0.0001
        )

        plot_val_data, _ = next(val_gen_f)
        plot_losses = PlotLosses(
            PLOT_FN,
            plot_val_data,
            STEPS_PER_EPOCH,
            input_3d=USE_3D_INPUT,
            layer=1
        )

        checkpoint = ModelCheckpoint(
            filepath=CHECKPOINT_FN,
            monitor='val_loss',
            verbose=VERBOSE,
            save_best_only=True,
            mode='min',
            save_weights_only=True,
            period=10
        )

        history = hg.fit_generator(
            generator=train_gen_f,
            steps_per_epoch=STEPS_PER_EPOCH,
            validation_data=val_gen_f,
            validation_steps=VALIDATION_STEPS,
            epochs=NUM_EPOCHS,
            verbose=VERBOSE,
            callbacks=[reduce_lr, plot_losses, checkpoint]
        )

        hg.save_weights(WEIGHTS_FN)
        with open(HISTORY_FN, "wb") as h:
            pk.dump(history.history, h)

    else:
        hg = load_model(ARCHITECTURE_FN, WEIGHTS_FN)

    if GENERATE_FEATURE_COUNTS:
        hg_t = transfer_learn_counts(
            hg,
            num_hg_modules=NUM_HG_MODULES,
            num_conv_channels=NUM_CONV_CHANNELS,
            transpose_output=GENERATE_SEGMENTED_CELLS,
            min_hg_shape=MIN_HG_SHAPE,
            max_hg_shape=MAX_HG_SHAPE,
            resampling_size=RESAMPLING_SIZE,
            learning_rate=LEARNING_RATE,
        )
        train_gen_c = cell_counting_wrapper(train_gen)
        val_gen_c = cell_counting_wrapper(val_gen)

        checkpoint_t = ModelCheckpoint(
            filepath=TRANSFER_CHECKPOINT_FN,
            monitor='val_loss',
            verbose=VERBOSE,
            save_best_only=True,
            mode='min',
            save_weights_only=True,
            period=10
        )

        history_t = hg_t.fit_generator(
            generator=train_gen_c,
            steps_per_epoch=STEPS_PER_EPOCH,
            validation_data=val_gen_c,
            validation_steps=VALIDATION_STEPS,
            epochs=NUM_TRANSFER_EPOCHS,
            verbose=VERBOSE,
            callbacks=[checkpoint_t, ]
        )

        hg_c = combine_models(hg, hg_t, LEARNING_RATE)

        model_json = hg_c.to_json()
        with open(COMBINED_ARCHITECTURE_FN, "w") as j:
            j.write(model_json)

        hg_c.save_weights(COMBINED_WEIGHTS_FN)

        with open(TRANSFER_HISTORY_FN, "wb") as h:
            pk.dump(history_t.history, h)
