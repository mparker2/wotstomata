import os
from glob import glob
import itertools as it
from PIL import Image

import numpy as np
from skimage.transform import resize

import click
import h5py
from h5py.version import hdf5_version_tuple

from ..train.hourglass import load_model
from ..predict.raster_to_shapes import (write_geoms_as_rois,
                                        edgemap_segments_to_polygons,
                                        heatmap_peaks_to_points)

# /fastdata does not support file locking
if hdf5_version_tuple[0] >= 1 and hdf5_version_tuple[1] >= 10:
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'


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


def _predict_img(zstack, model, step_size=128):
    assert zstack.shape[-1] == 3
    heatmap = []
    segments = []
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
        hm_preds, seg_preds, *_, blur = model.predict(
            np.asarray(X))[-5:]
        z_hm = np.zeros((imax, jmax), dtype='f')
        z_segs = z_hm.copy()
        z_blur = z_hm.copy()
        tot_in_pos = z_hm.copy()
        for (i, j), h, s, b in zip(pos, hm_preds, seg_preds, blur.ravel()):
            h = resize(threshold_image(h).reshape(h.shape[:-1]),
                       (256, 256), preserve_range=True)
            s = resize(threshold_image(s).reshape(s.shape[:-1]),
                       (256, 256), preserve_range=True)
            z_hm[i: i + 256, j: j + 256] += h[:min(h.shape[0], imax - i),
                                              :min(h.shape[1], jmax - j)]
            z_segs[i: i + 256, j: j + 256] += s[:min(s.shape[0], imax - i),
                                                :min(s.shape[1], jmax - j)]
            z_blur[i: i + 256, j: j + 256] += b
            tot_in_pos[i:i + 256, j: j + 256] += 1
        heatmap.append(z_hm / tot_in_pos)
        segments.append(z_segs / tot_in_pos)
        blur_levels.append(z_blur / tot_in_pos)
    return np.asarray(heatmap), np.asarray(segments), np.asarray(blur_levels)


@click.command()
@click.option('--img', required=True,
              help='zstack tiff to predict on, or directory of tifs')
@click.option('--arch', required=True, help='json model architecture')
@click.option('--weights', required=True, help='h5 model weights')
@click.option('--output', required=False, default=None,
              help='otuput file basename')
@click.option('--step-size', default=128)
@click.option('--include-inverted/--no-inverted', default=True)
@click.option('--generate-rois/--no-rois', default=True)
@click.option('--roi-lines/--roi-polys', default=False)
@click.option('--points-3d/--points-flattened', default=False)
def predict_image(img, arch, weights, output,
                  step_size, include_inverted,
                  generate_rois, roi_lines, points_3d):
    model = load_model(arch, weights)
    if os.path.isdir(img):
        glob_path = img.rstrip('/') + '/*.tif'
        imgs = glob(glob_path)
        outputs = [os.path.splitext(i)[0] for i in imgs]
    else:
        imgs = [img, ]
        outputs = [output, ]
    for img_fn, output in zip(imgs, outputs):
        img = read_multipage_tiff(img_fn)
        heatmap, segments, blur = _predict_img(
            img, model, step_size)
        h5_file = h5py.File(output, mode='w')
        h5_file.create_dataset('heatmap', data=heatmap)
        h5_file.create_dataset('segments', data=segments)
        h5_file.create_dataset('blur', data=blur)
        if include_inverted:
            heatmap_i, segments_i, blur_i = _predict_img(
                img[:, ::-1, ::-1, :], model, step_size)
            heatmap_i = heatmap_i[:, ::-1, ::-1]
            segments_i = segments_i[:, ::-1, ::-1]
            blur_i = blur_i[:, ::-1, ::-1]
            h5_file = h5py.File(output + '.prediction.h5', mode='w')
            h5_file.create_dataset('heatmap_i', data=heatmap_i)
            h5_file.create_dataset('segments_i', data=segments_i)
            h5_file.create_dataset('blur_i', data=blur_i)
        h5_file.close()
        if generate_rois:
            if include_inverted:
                heatmap = np.concatenate([heatmap, heatmap_i], axis=0)
                segments = np.concatenate([segments, segments_i], axis=0)
                blur = np.concatenate([blur, blur_i], axis=0)
            i, j = np.indices(segments.shape[1:-1])
            seg_max = segments[np.argmax(blur, axis=0), i, j]
            polys = edgemap_segments_to_polygons(seg_max)
            if not points_3d:
                heatmap = heatmap[np.argmax(blur, axis=0), i, j]
            points = heatmap_peaks_to_points(heatmap)
            write_geoms_as_rois(polys, output + '.segmentedcells.roi')
            write_geoms_as_rois(points, output + '.stomatapoints.roi')

if __name__ == '__main__':
    predict_image()
