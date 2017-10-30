import warnings
import random
from glob import glob

from PIL import Image

import numpy as np
import pandas as pd

from skimage.transform import resize
from skimage.filters import gaussian
from skimage.morphology import disk

from keras.preprocessing.image import ImageDataGenerator


def pad_to_size(arr, s):
    sh, sw = s
    ah, aw = arr.shape
    h_pad, hr = divmod(sw - aw, 2)
    w_pad, wr = divmod(sh - ah, 2)
    padded =  np.pad(
        arr,
        ((h_pad, h_pad + hr), (w_pad, w_pad + wr)),
        'constant',
        constant_values=0)
    return padded


def get_gaussian(sigma, truncate=3):
    s = int(sigma * (truncate + 1)) * 2 + 1
    m = s // 2
    g = np.zeros((s, s))
    g[m, m] = 1
    g = gaussian(g, sigma)
    trunc = pad_to_size(disk(sigma * truncate), g.shape).astype(bool)
    g[~trunc] = 0
    return g, m


def create_heatmap(img, points, sigma, trunc):
    gaus = np.zeros(img.shape[:-1])
    vals = zip(points.Y.values.round().astype('i'),
               points.X.values.round().astype('i'),
               points.Slice.values.astype('i'))
    for x, y, z in vals:
        z_img = img[z]
        z_gaus = gaus[z]
        g, m = get_gaussian(sigma, trunc)
        i_slic = slice(max(x - m, 0),
                       min(x + m + 1, z_img.shape[0]))
        j_slic = slice(max(y - m, 0),
                       min(y + m + 1, z_img.shape[1]))
        if x - m < 0:
            g = g[abs(x - m):]
        if y - m < 0:
            g = g[:, abs(y - m):]
        if x + m + 1 > z_img.shape[0]:
            g = g[:np.negative(x + m + 1 - z_img.shape[0])]
        if y + m + 1 > z_img.shape[1]:
            g = g[:, :np.negative(y + m + 1 - z_img.shape[1])]
        sect = z_gaus[i_slic, j_slic].copy()
        z_gaus[i_slic, j_slic] = np.max(np.asarray([sect, g]), axis=0)
    return img, gaus / gaus.max()


def recentre_point(i, img_size):
    l = - min(i - 128, 0)
    r = img_size - max(i + 128, img_size)
    return i + l + r

def get_indices(points):
    points = points.sample(frac=1)
    x = points.X.values
    y = points.Y.values
    z = points.Slice.values
    determined = zip(y.round().astype('i'),
                     x.round().astype('i'),
                     z.astype('i'))
    return determined

def subset_img(img, hm, points, frac_random):
    determined = get_indices(points)
    while True:
        if np.random.random() > frac_random:
            try:
                i, j, n = next(determined)
            except StopIteration:
                determined = get_indices(points)
                i, j, n = next(determined)
            z_img = img[n]
            z_hm = hm[n]
            i = recentre_point(i, z_img.shape[0])
            j = recentre_point(j, z_img.shape[1])
            i_slic = slice(i - 128, i + 128)
            j_slic = slice(j - 128, j + 128)
            img_slic = z_img[i_slic, j_slic, :]
            hm_slic = z_hm[i_slic, j_slic]
        else:
            z = np.random.randint(0, img.shape[0])
            i = np.random.randint(128, img.shape[1] - 128)
            j = np.random.randint(128, img.shape[2] - 128)
            i_slic = slice(i - 128, i + 128)
            j_slic = slice(j - 128, j + 128)
            img_slic = img[z, i_slic, j_slic, :]
            hm_slic = hm[z, i_slic, j_slic]
        yield img_slic, hm_slic


def read_multipage_tiff(path):
    img = Image.open(path)
    n_bands = len(img.getbands())
    img_shape = img.height, img.width
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        z = np.asarray(img.getdata())
        z = z.reshape((*img_shape, n_bands))
        images.append(z.astype(np.uint8))
    img.close()
    images = np.asarray(images)
    return images


def preprocess_img(img_fn, point_fn,
                   sigma=20, trunc=3,
                   frac_random=0.5):
    points = pd.read_csv(point_fn)
    points.Slice -= 1 # imageJ slices are 1-indexed
    img = read_multipage_tiff(img_fn)
    img, hmap = create_heatmap(img, points, sigma, trunc)
    return img, hmap, points


def resize_y(y_batch, resize_shape):
    resized_y_batch = []
    for hm in y_batch:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            resized_y_batch.append(resize(hm, resize_shape))
    resized_y_batch = np.asarray(resized_y_batch)
    return resized_y_batch


def generate_training_data(directory,
                           batch_size,
                           num_hg_modules,
                           y_resize_shape=(64, 64, 1)):
    
    all_img_generators = []
    for img_fn in glob(directory + '/*.tif'):
        img, heatmap, points = preprocess_img(img_fn, img_fn + '.csv')
        all_img_generators.append(
            subset_img(img, heatmap, points, frac_random=0.5))
    
    image_transformer = ImageDataGenerator(
        rotation_range=180,
        fill_mode='constant',
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True
        )

    while True:
        X_batch = []
        y_batch = []
        for _ in range(batch_size):
            gen = random.choice(all_img_generators)
            seed = np.random.randint(0, 1000)
            X, y = next(gen)
            y.shape = y.shape + (1, )
            X = image_transformer.random_transform(X, seed=seed)
            y = image_transformer.random_transform(y, seed=seed)
            X_batch.append(X)
            y_batch.append(y)
        resized_y_batch = resize_y(y_batch, y_resize_shape)
        X_batch = np.asarray(X_batch)
        yield (X_batch, [resized_y_batch, ] * num_hg_modules)
