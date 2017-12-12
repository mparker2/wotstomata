import pickle as pk

from keras.callbacks import ReduceLROnPlateau

from hourglass import build_hourglass
from img_generator import generate_training_data
from plotting import PlotLosses


NUM_HG_MODULES = 4
NUM_CONV_CHANNELS = 16
MIN_SHAPE = 4
LEARNING_RATE = 2.5e-4
MINIBATCH_SIZE = 32
FRAC_RANDOMLY_SAMPLED_IMGS = 0.4
STEPS_PER_EPOCH = 128
VALIDATION_STEPS = 12
NUM_EPOCHS = 250


if __name__ == '__main__':

    train_gen = generate_training_data(
        '/fastdata/mbp14mtp/stomatal_prediction/train',
        batch_size=MINIBATCH_SIZE,
        num_hg_modules=NUM_HG_MODULES,
        segment=True,
        frac_randomly_sampled_imgs=FRAC_RANDOMLY_SAMPLED_IMGS)

    val_gen = generate_training_data(
        '/fastdata/mbp14mtp/stomatal_prediction/val',
        batch_size=MINIBATCH_SIZE,
        num_hg_modules=NUM_HG_MODULES,
        segment=True,
        frac_randomly_sampled_imgs=FRAC_RANDOMLY_SAMPLED_IMGS)

    hg = build_hourglass(num_hg_modules=NUM_HG_MODULES,
                         num_conv_channels=NUM_CONV_CHANNELS,
                         min_shape=MIN_SHAPE,
                         transpose_output=True,
                         learning_rate=LEARNING_RATE)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.1,
                                  patience=20,
                                  cooldown=20,
                                  verbose=1,
                                  epsilon=0.0001)

    plot_val_data, _ = next(val_gen)
    plot_losses = PlotLosses("./segmentation_loss.png",
                             plot_val_data,
                             STEPS_PER_EPOCH,
                             layer=1)

    history = hg.fit_generator(
        generator=train_gen,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=val_gen,
        validation_steps=VALIDATION_STEPS,
        epochs=NUM_EPOCHS,
        verbose=True,
        callbacks=[reduce_lr, plot_losses]
    )

    model_json = hg.to_json()
    with open("segmentation_model_architecture.json", "w") as j:
        j.write(model_json)
    hg.save_weights("segmentation_model_weights.h5")
    with open("segmentation_model_history.pk", "wb") as h:
        pk.dump(history.history, h)
