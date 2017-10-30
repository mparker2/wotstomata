import pickle as pk

from keras.callbacks import ReduceLROnPlateau

from hourglass import build_hourglass
from img_generator import generate_training_data
from plotting import PlotLosses


NUM_HG_MODULES = 4
NUM_CONV_CHANNELS = 16
MIN_SHAPE = 4
MINIBATCH_SIZE = 32
STEPS_PER_EPOCH = 128


if __name__ == '__main__':

    train_gen = generate_training_data(
        '/fastdata/mbp14mtp/stomatal_prediction/train',
        batch_size=MINIBATCH_SIZE,
        num_hg_modules=NUM_HG_MODULES,
        y_resize_shape=(64, 64, 1))


    val_gen = generate_training_data(
        '/fastdata/mbp14mtp/stomatal_prediction/val',
        batch_size=MINIBATCH_SIZE,
        num_hg_modules=NUM_HG_MODULES,
        y_resize_shape=(64, 64, 1))

    hg = build_hourglass(num_hg_modules=NUM_HG_MODULES,
                         num_conv_channels=NUM_CONV_CHANNELS,
                         min_shape=MIN_SHAPE)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.1,
                                  patience=10,
                                  cooldown=10,
                                  verbose=1,
                                  epsilon=0.001)

    plot_val_data, _ = next(val_gen)
    plot_losses = PlotLosses("./training_loss.png",
                             plot_val_data,
                             STEPS_PER_EPOCH)

    history = hg.fit_generator(
        generator=train_gen,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=val_gen,
        validation_steps=2,
        epochs=100,
        verbose=False,
        callbacks=[reduce_lr, plot_losses]
    )

    model_json = hg.to_json()
    with open("model_architecture.json", "w") as j:
        j.write(model_json)
    hg.save_weights("model_weights.h5")
    with open("model_history.pk", "wb") as h:
        pk.dump(history.history, h)
