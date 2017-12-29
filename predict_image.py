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
    stomatal_density = []
    all_cell_density = []
    blur_levels = []
    zmax, imax, jmax, _ = zstack.shape
    for z in range(zmax):
        pos = []
        X = []
        z_slic = slice(max(0, z - 1), min(z + 2, zmax))
        for i, j in it.product(range(0, imax, step_size),
                               range(0, jmax, step_size)):
            slic = zstack[z_slic, i: i + 256, j: j + 256, :]
            if not slic.shape == (3, 256, 256, 3):
                padded = np.zeros((3, 256, 256, 3))
                padded[:slic.shape[0],
                       :slic.shape[1],
                       :slic.shape[2],
                       ...] = slic
                slic = padded
            X.append(slic)
            pos.append((i, j))
        hm_preds, seg_preds, stomata, all_cells, blur = model.predict(
            np.asarray(X))[-5:]
        z_hm = np.zeros((imax, jmax), dtype='f')
        z_segs = z_hm.copy()
        z_density = z_hm.copy()
        z_all_cells = z_hm.copy()
        z_blur = z_hm.copy()
        tot_in_pos = z_hm.copy()
        for (i, j), h, s, st, a, b in zip(pos, hm_preds, seg_preds,
                                          stomata.ravel(), all_cells.ravel(),
                                          blur.ravel()):
            h = resize(threshold_image(h).reshape(h.shape[:-1]),
                       (256, 256), preserve_range=True)
            s = resize(threshold_image(s).reshape(s.shape[:-1]),
                       (256, 256), preserve_range=True)
            z_hm[i: i + 256, j: j + 256] += h[:min(h.shape[0], imax - i),
                                              :min(h.shape[1], jmax - j)]
            z_segs[i: i + 256, j: j + 256] += s[:min(s.shape[0], imax - i),
                                                :min(s.shape[1], jmax - j)]
            z_density[i: i + 256, j: j + 256] += st
            z_all_cells[i: i + 256, j: j + 256] += a
            z_blur[i: i + 256, j: j + 256] += b
            tot_in_pos[i:i + 256, j: j + 256] += 1
        heatmap.append(z_hm / tot_in_pos)
        segments.append(z_segs / tot_in_pos)
        stomatal_density.append(z_density / tot_in_pos)
        all_cell_density.append(z_all_cells / tot_in_pos)
        blur_levels.append(z_blur / tot_in_pos)
    blur_levels = np.asarray(blur_levels)
    return (np.asarray(heatmap), np.asarray(segments),
            np.asarray(stomatal_density),
            np.asarray(all_cell_density),
            np.asarray(blur_levels))


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
        heatmap, segments, density, all_cells, blur = _predict_img(
            img, model, step_size)
        h5_file = h5py.File(output, mode='w')
        h5_file.create_dataset('heatmap', data=heatmap)
        h5_file.create_dataset('segments', data=segments)
        h5_file.create_dataset('stomatal_density', data=density)
        h5_file.create_dataset('all_cell_density', data=all_cells)
        h5_file.create_dataset('blur', data=blur)
        h5_file.close()

if __name__ == '__main__':
    predict_img()
