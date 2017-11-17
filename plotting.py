import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras.callbacks import Callback

sns.set_color_codes()
plt.switch_backend('agg')


class PlotLosses(Callback):

    def __init__(self, file_name, val_data, steps_per_epoch, layer=0):
        super(PlotLosses, self).__init__()
        self.file_name = file_name
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
        self.img_ax.imshow(self.val_data[i])
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
        self.img_ax.imshow(self.val_data[i], extent=[0, 1, 0, 1], cmap='gray')
        self.img_ax.imshow(pred, extent=[0, 1, 0, 1], cmap='viridis', alpha=0.3)
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
