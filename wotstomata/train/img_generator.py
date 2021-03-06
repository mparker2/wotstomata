import os
import random
import warnings
from glob import glob
from itertools import repeat

from PIL import Image

import numpy as np
from scipy import ndimage as ndi
import pandas as pd

from skimage.transform import resize
from skimage.filters import gaussian
from skimage.morphology import disk, remove_small_objects
from skimage.measure import label
from skimage.feature import peak_local_max
from skimage.draw import line as draw_line, polygon as draw_polygon
from skimage.segmentation import clear_border, relabel_sequential

from shapely.geometry import Point
from shapely.ops import cascaded_union

from keras.preprocessing.image import ImageDataGenerator

import read_roi


def pad_to_size(arr, s):
    sh, sw = s
    ah, aw = arr.shape
    h_pad, hr = divmod(sw - aw, 2)
    w_pad, wr = divmod(sh - ah, 2)
    padded = np.pad(
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


def add_multipart_line(img, line_x, line_y):
    coords_iter = zip(line_y, line_x)
    r0, c0 = next(coords_iter)
    for r1, c1 in coords_iter:
        rr, cc = draw_line(r0 - 1, c0 - 1, r1 - 1, c1 - 1)
        img[rr, cc] = 1
        r0, c0 = r1, c1
    return img


def roi_zip_to_segment_image(roi_zip_fn, img_shape, dilation_size):
    rois = read_roi.read_roi_zip(roi_zip_fn)
    drawing = np.zeros(img_shape)
    for line in rois.values():
        drawing = add_multipart_line(drawing, line['x'], line['y'])
    drawing = ndi.binary_dilation(drawing, structure=disk(dilation_size))
    drawing = drawing.astype('f')
    return drawing


def filter_segments_by_z_position(segs, points, num_z_pos,
                                  buffer_size=100, gaus_sigma=50):
    masked_segs = np.tile(segs.reshape(1, *segs.shape),
                          (num_z_pos, 1, 1))
    masked_region = []
    grouped = points.groupby('Slice')
    for z in range(num_z_pos):
        try:
            zp = grouped.get_group(z)
        except KeyError:
            masked_segs[z] = 0
            masked_region.append(np.ones_like(segs))
            continue
        z_points = [Point(x, y) for _, x, y in zp[['X', 'Y']].itertuples()]
        in_focus_region = (cascaded_union(z_points).convex_hull
                                                   .buffer(buffer_size))
        in_focus_mask = np.ones_like(segs)
        c, r = zip(*in_focus_region.exterior.coords)
        r_idx, c_idx = draw_polygon(r, c, segs.shape)
        in_focus_mask[r_idx, c_idx] = 0
        in_focus_mask_g = gaussian(in_focus_mask,
                                   sigma=50,
                                   preserve_range=True)
        masked_segs[z] -= in_focus_mask_g
        masked_region.append(in_focus_mask)
    masked_segs[masked_segs < 0] = 0
    masked_region = 1 - np.asarray(masked_region)
    return np.asarray(masked_segs), masked_region


def preprocess_img(img_fn, point_fn, roi_fn=None, sigma=20, trunc=3):
    points = pd.read_csv(point_fn)
    points.Slice -= 1  # imageJ slices are 1-indexed
    img = read_multipage_tiff(img_fn)
    img, hmap = create_heatmap(img, points, sigma, trunc)
    if roi_fn is not None:
        segs = roi_zip_to_segment_image(roi_fn,
                                        img.shape[1:3],
                                        dilation_size=2)
        segs, mask = filter_segments_by_z_position(segs, points, img.shape[0])
    else:
        segs = None
        mask = None
    return img, hmap, points, segs, mask


def pad_3d_img(img, pad_left=True):
    padding = np.zeros((1, 256, 256, 3))
    to_concat = [padding, img] if pad_left else [img, padding]
    padded = np.concatenate(to_concat, axis=0)
    return padded


def iter_img_subsets(img, hm, points, segs=None, mask=None,
                     frac_random=0.5, input_3d=False):
    determined = get_indices(points)
    while True:
        if np.random.random() > frac_random:
            try:
                i, j, z = next(determined)
            except StopIteration:
                determined = get_indices(points)
                i, j, z = next(determined)
            i = recentre_point(i, img.shape[1])
            j = recentre_point(j, img.shape[2])
        else:
            z = np.random.randint(0, img.shape[0])
            i = np.random.randint(128, img.shape[1] - 128)
            j = np.random.randint(128, img.shape[2] - 128)
        i_slic = slice(i - 128, i + 128)
        j_slic = slice(j - 128, j + 128)

        z_slic = z if not input_3d else slice(
            max(0, z - 1),
            min(z + 2, img.shape[0])
        )
        img_slic = img[z_slic, i_slic, j_slic]
        hm_slic = hm[z_slic, i_slic, j_slic]
        segs_slic = segs[z_slic, i_slic, j_slic] if segs is not None else None
        mask_slic = mask[z_slic, i_slic, j_slic] if mask is not None else None
        if input_3d:
            if img_slic.shape[0] < 3:
                img_slic = pad_3d_img(img_slic, pad_left=not z)
            hm_slic = hm_slic.max(0)
            segs_slic = segs_slic.max(0)
            mask_slic = mask_slic.max(0)
        img_slic = img_slic.astype(np.ubyte)
        yield img_slic, hm_slic, segs_slic, mask_slic


def get_all_img_generators(directory,
                           segment,
                           frac_randomly_sampled_imgs,
                           input_3d):
    all_img_generators = []
    for img_fn in glob(directory + '/*.tif'):
        if os.path.exists(img_fn + '.csv'):
            if segment and not os.path.exists(img_fn + '.roi.zip'):
                continue
            img, heatmap, points, segs, mask = preprocess_img(
                img_fn, img_fn + '.csv', img_fn + '.roi.zip')
            all_img_generators.append(
                iter_img_subsets(
                    img, heatmap, points, segs, mask,
                    frac_random=frac_randomly_sampled_imgs,
                    input_3d=input_3d))
    return all_img_generators


def resize_y_batch(y_batch, shape):
    y_batch = np.asarray(y_batch)
    if y_batch.ndim == 4 and y_batch.shape[-1] == 1:
        y_batch.shape = y_batch.shape[:-1]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        y_batch_resized = np.asarray(
            [resize(img, shape, preserve_range=True) for img in y_batch])
    if y_batch_resized.ndim == 3:
        y_batch_resized.shape = y_batch_resized.shape + (1, )
    return y_batch_resized


def loop_training_iterators(iterators,
                            transformer,
                            batch_size,
                            segment):
    while True:
        X_batch = []
        y_hm_batch = []
        if segment:
            y_segs_batch = []
            y_mask_batch = []
        else:
            y_segs_batch = None
            y_mask_batch = None
        for _ in range(batch_size):
            gen = random.choice(iterators)
            seed = np.random.randint(0, 10000)
            X, y_hm, y_segs, y_mask = next(gen)
            y_hm.shape = y_hm.shape + (1, )
            X = transformer(X, seed=seed)
            y_hm = transformer(y_hm, seed=seed)
            X_batch.append(X)
            y_hm_batch.append(y_hm)
            if segment:
                y_segs.shape = y_segs.shape + (1, )
                y_segs = transformer(y_segs, seed=seed)
                y_mask.shape = y_mask.shape + (1, )
                y_mask = transformer(y_mask, seed=seed)
                y_segs_batch.append(y_segs)
                y_mask_batch.append(y_mask)
        X_batch = np.asarray(X_batch)
        yield X_batch, y_hm_batch, y_segs_batch, y_mask_batch


def transform_3d(transformer):
    def _transform_3d(zstack, seed):
        if zstack.ndim == 4:
            transformed = []
            for z in zstack:
                transformed.append(transformer(z, seed))
            return np.asarray(transformed)
        else:
            return transformer(zstack, seed)
    return _transform_3d


def generate_training_data(directory,
                           batch_size,
                           segment=True,
                           frac_randomly_sampled_imgs=0.5,
                           input_3d=True):
    all_img_generators = get_all_img_generators(
        directory,
        segment,
        frac_randomly_sampled_imgs,
        input_3d)

    image_transformer = ImageDataGenerator(
        rotation_range=180,
        fill_mode='constant',
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)

    if not input_3d:
        transformer = image_transformer.random_transform
    else:
        transformer = transform_3d(image_transformer.random_transform)

    yield from loop_training_iterators(
        all_img_generators,
        transformer,
        batch_size,
        segment)


def transformed_fill_mask(img_batch):
    if img_batch.ndim == 5:
        transformed_fill_mask_batch = img_batch[:, 1, ...].any(axis=3)
    else:
        transformed_fill_mask_batch = img_batch.any(axis=3)
    return transformed_fill_mask_batch


def batch_sum(batch):
    batch = np.asarray(batch)
    flat_batch = batch.reshape(batch.shape[0], -1)
    return flat_batch.sum(1)


def estimate_in_focus(in_focus_batch, transformed_fill_mask_batch):
    in_focus_count = batch_sum(in_focus_batch)
    img_sizes = batch_sum(transformed_fill_mask_batch)
    return in_focus_count / img_sizes


def batch_formatting_wrapper(generator,
                             n_outputs,
                             heatmap_shape=(64, 64),
                             segment=True,
                             segment_shape=(128, 128),
                             in_focus=True):
    for X_batch, y_hm_batch, y_segs_batch, in_focus_mask_batch in generator:
        y_hm_batch = resize_y_batch(y_hm_batch, heatmap_shape)
        if segment:
            y_segs_batch = resize_y_batch(y_segs_batch, segment_shape)
            y_batch = [y_hm_batch, y_segs_batch] * n_outputs
        else:
            y_batch = [y_hm_batch, ] * n_outputs
        if in_focus:
            transformed_fill_mask_batch = transformed_fill_mask(X_batch)
            in_focus_score_batch = estimate_in_focus(
                in_focus_mask_batch,
                transformed_fill_mask_batch)
            y_batch += [in_focus_score_batch, ]
        yield X_batch, y_batch
