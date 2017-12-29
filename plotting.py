import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

from skimage import img_as_float
from skimage.exposure import adjust_gamma

from keras.callbacks import Callback

if 'DISPLAY' not in os.environ:
    plt.switch_backend('agg')

sns.set_color_codes()


def false_colour(img, probabilities,
                 img_gain=0.5, prob_gain=0.5,
                 cmap='viridis'):
    bw = img_as_float(img)
    colours = cm.get_cmap(cmap)(probabilities)[..., :3]
    if img_gain is not None:
        bw = adjust_gamma(bw, gain=img_gain)
    if prob_gain is not None:
        colours = adjust_gamma(colours, gain=prob_gain)
    false_coloured = bw + colours
    return false_coloured


class PlotLosses(Callback):

    def __init__(self, file_name, val_data,
                 steps_per_epoch, input_3d, layer=0):
        super(PlotLosses, self).__init__()
        self.file_name = file_name
        self.input_is_3d = input_3d
        self.val_data = val_data
        self.batch_steps = np.linspace(0, 1, steps_per_epoch, endpoint=False)
        self.epoch = 0
        self.layer = 0

    def on_train_begin(self, logs={}):
        self.train_losses = [np.nan, ]
        self.batch_train_losses = []
        self.batch_train_xvals = []
        self.val_losses = [np.nan, ]

        i = np.random.randint(0, self.val_data.shape[0])
        self.fig, axes = plt.subplots(figsize=(10, 4), ncols=2)
        self.loss_ax = axes[0]
        self.loss_ax.set_ylim(0, 0.1)
        self.img_ax = axes[1]
        img = self.val_data[i] if not self.input_is_3d else self.val_data[i][1]
        self.img_ax.imshow(img, extent=[0, 1, 0, 1], cmap='gray')
        self.img_ax.set_axis_off()
        plt.tight_layout()

    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1
        self.train_losses.append(logs.get('loss', np.nan))
        self.val_losses.append(logs.get('val_loss', np.nan))

        self.loss_ax.clear()
        self.loss_ax.plot(
            self.batch_train_xvals,
            self.batch_train_losses,
            color='b',
            label='batch train',
            ls='--')
        self.loss_ax.plot(self.train_losses, 'bo-', label='train')
        self.loss_ax.plot(self.val_losses, 'go-', label='validation')
        plt.legend()

        i = np.random.randint(0, self.val_data.shape[0])
        pred = self.model.predict(np.asarray([self.val_data[i], ]))[3][0]
        pred = pred[:, :, self.layer]

        self.img_ax.clear()
        img = self.val_data[i] if not self.input_is_3d else self.val_data[i][1]
        self.img_ax.imshow(img, extent=[0, 1, 0, 1], cmap='gray')
        self.img_ax.imshow(pred,
                           extent=[0, 1, 0, 1],
                           cmap='viridis',
                           alpha=0.3)
        self.img_ax.set_axis_off()
        plt.savefig(self.file_name)

    def on_batch_end(self, batch, logs={}):
        self.batch_train_losses.append(logs.get('loss', np.nan))
        self.batch_train_xvals.append(self.epoch + self.batch_steps[batch])

        self.loss_ax.clear()
        self.loss_ax.plot(
            self.batch_train_xvals,
            self.batch_train_losses,
            color='b',
            ls='--')
        self.loss_ax.plot(self.train_losses, 'bo-')
        self.loss_ax.plot(self.val_losses, 'go-')
        plt.savefig(self.file_name)
