import itertools as it
from PIL import Image

import numpy as np

from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.transform import resize

from sklearn.metrics import roc_curve

from keras.models import model_from_json

from shapely.geometry import Point
from shapely.ops import cascaded_union


def load_model(model_json_fn, model_weights_fn):
    with open(model_json_fn) as j:
        model = model_from_json(j.read())
    model.load_weights(model_weights_fn)
    return model


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


def predict_img(img_fn, model, step=128):
    zstack = read_multipage_tiff(img_fn)
    output = []
    for z in zstack:
        imax, jmax, _ = z.shape
        pos = []
        X = []
        for i, j in it.product(range(0, imax, step),
                               range(0, jmax, step)):
            slic = z[i: i + 256, j: j + 256, :]
            if not slic.shape == (256, 256, 3):
                padded = np.zeros((256, 256, 3))
                padded[:slic.shape[0], :slic.shape[1], :] = slic
                slic = padded
            X.append(slic)
            pos.append((i, j))
        preds = model.predict(np.asarray(X))[3]
        z_output = np.zeros(z.shape[:-1], dtype='f')
        tot_in_pos = np.zeros(z.shape[:-1], dtype='i')
        for (i, j), p in zip(pos, preds):
            p[p < 0] = 0
            p[p > 1] = 1
            p = resize(p.reshape(64, 64), (256, 256))
            z_output[i: i + 256, j: j + 256] += p[:min(p.shape[0], imax - i),
                                                  :min(p.shape[1], jmax - j)]
            tot_in_pos[i:i + 256, j: j + 256] += 1
        output.append(z_output / tot_in_pos)
    return np.asarray(output)


def get_stomatal_positions(heatmap,
                           sigma,
                           min_distance,
                           exclude_border):
    max_pred = np.max(heatmap, axis=0)
    max_pred = gaussian(max_pred, sigma)
    maxima = peak_local_max(max_pred,
                            min_distance=min_distance,
                            exclude_border=exclude_border)
    vals = np.asarray([max_pred[x, y] for x, y in maxima])
    return [Point(x, y) for y, x in maxima], vals


def get_flattened_true_data(y_true, x_col, y_col, buffer_size):
    points = [Point(x, y) for _, x, y in y_true[[x_col, y_col]].itertuples()]
    buffered_points = [point.buffer(buffer_size) for point in points]
    flattened = [poly for poly in cascaded_union(buffered_points).geoms]
    centroids = [poly.centroid for poly in flattened]
    return centroids


def filter_true_at_borders(y_true, img_shape, exclude_border):
    if exclude_border:
        filtered_y_true = []
        for p in y_true:
            if exclude_border < p.x < (img_shape[0] - exclude_border):
                if exclude_border < p.y < (img_shape[1] - exclude_border):
                    filtered_y_true.append(p)
    return filtered_y_true


def precision_score_positional(buffered_y_true, y_pred):
    tp = 0
    for p in y_pred:
        is_true = any([p.within(poly) for poly in buffered_y_true])
        if is_true:
            tp += 1
    try:
        precision = tp / float(len(y_pred))
    except ZeroDivisionError:
        precision = 0
    return precision


def recall_score_positional(buffered_y_true, y_pred):
    tp = 0
    for poly in buffered_y_true:
        is_predicted = any([p.within(poly) for p in y_pred])
        if is_predicted:
            tp += 1
    recall = tp / float(len(buffered_y_true))
    return recall


def f1_score_positional(y_true, y_pred,
                        buffer_size, img_shape, exclude_border):
    y_true = filter_true_at_borders(y_true, img_shape, exclude_border)
    buffered_y_true = [p.buffer(buffer_size) for p in y_true]
    precision = precision_score_positional(buffered_y_true, y_pred)
    recall = recall_score_positional(buffered_y_true, y_pred)
    try:
        f1_score = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1_score = 0
    return f1_score


def precision_recall_curve(y_true, y_pred, y_pred_vals,
                           buffer_size, img_shape, exclude_border):
    y_true = filter_true_at_borders(y_true, img_shape, exclude_border)
    buffered_y_true = [p.buffer(buffer_size) for p in y_true]
    precision = []
    recall = []
    for t in np.linspace(0, 1, 500):
        y_pred_filtered = [p for p, v in zip(y_pred, y_pred_vals) if v > t]
        if not y_pred_filtered:
            continue
        precision.append(precision_score_positional(buffered_y_true,
                                                    y_pred_filtered))
        recall.append(recall_score_positional(buffered_y_true,
                                              y_pred_filtered))
    return np.asarray(precision), np.asarray(recall)
