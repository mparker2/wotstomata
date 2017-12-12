import os
from glob import glob
import itertools as it
from PIL import Image

import numpy as np
from skimage.transform import resize

import click
import h5py

from hourglass import load_model


def read_multipage_tiff(path):
    img = Image.open(path)
    n_bands = len(img.getbands())
    img_shape = img.height, img.width
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        z = np.asarray(img.getdata())
        z = z.reshape(img_shape + (n_bands, ))
        images.append(z.astype(np.uint8))
    img.close()
    images = np.asarray(images)
    return images


def threshold_image(img):
    img[img > 1] = 1
    img[img < 0] = 0
    return img


def _predict_img(img_fn, model, step_size=128):
    zstack = read_multipage_tiff(img_fn)
    assert zstack.shape[-1] == 3
    heatmap = []
    segments = []
    for z in zstack:
        imax, jmax, _ = z.shape
        pos = []
        X = []
        for i, j in it.product(range(0, imax, step_size),
                               range(0, jmax, step_size)):
            slic = z[i: i + 256, j: j + 256, :]
            if not slic.shape == (256, 256, 3):
                padded = np.zeros((256, 256, 3))
                padded[:slic.shape[0], :slic.shape[1], :] = slic
                slic = padded
            X.append(slic)
            pos.append((i, j))
        hm_preds, seg_preds = model.predict(np.asarray(X))[-2:]
        z_hm = np.zeros(z.shape[:-1], dtype='f')
        z_segs = np.zeros(z.shape[:-1], dtype='f')
        tot_in_pos = np.zeros(z.shape[:-1], dtype='i')
        for (i, j), h, s, in zip(pos, hm_preds, seg_preds):
            h = resize(threshold_image(h).reshape(64, 64),
                       (256, 256), preserve_range=True)
            s = resize(threshold_image(s).reshape(64, 64),
                       (256, 256), preserve_range=True)            
            z_hm[i: i + 256, j: j + 256] += h[:min(h.shape[0], imax - i),
                                              :min(h.shape[1], jmax - j)]
            z_segs[i: i + 256, j: j + 256] += s[:min(s.shape[0], imax - i),
                                                :min(s.shape[1], jmax - j)]
            tot_in_pos[i:i + 256, j: j + 256] += 1
        heatmap.append(z_hm / tot_in_pos)
        segments.append(z_segs / tot_in_pos)
    return np.asarray(heatmap), np.asarray(segments)


@click.command()
@click.option('--img', required=True,
              help='zstack tiff to predict on, or directory of tifs')
@click.option('--arch', required=True, help='json model architecture')
@click.option('--weights', required=True, help='h5 model weights')
@click.option('--output', required=False, default=None,
              help='otuput hdf5 file')
@click.option('--step-size', default=128)
def predict_img(img, arch, weights, output, step_size):
    model = load_model(arch, weights)
    if os.path.isdir(img):
        glob_path = img.rstrip('/') + '/*.tif'
        imgs = glob(glob_path)
        outputs = [os.path.splitext(i)[0] + '.prediction.h5' for i in imgs]
    else:
        imgs = [img, ]
        outputs = [output, ]
    for img, output in zip(imgs, outputs):
        heatmap, segments = _predict_img(img, model, step_size)
        h5_file = h5py.File(output, mode='w')
        h5_file.create_dataset('heatmap', data=heatmap)
        h5_file.create_dataset('segments', data=segments)
        h5_file.close()

if __name__ == '__main__':
    predict_img()
