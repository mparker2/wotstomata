import os
import zipfile
import shutil

import numpy as np

from skimage.segmentation import (checkerboard_level_set,
                                  morphological_chan_vese,
                                  relabel_sequential)
from skimage.measure import label
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage.future.graph import RAG

from rasterio.features import shapes as polygonize
from shapely.geometry import shape as as_shape, MultiPoint, LineString

from .roi import ROIEncoder, ROI_TYPES


def active_contour_segmentation(img):
    contours = checkerboard_level_set(img.shape, 2)
    contours = morphological_chan_vese(img, 50,
                                       init_level_set=contours,
                                       smoothing=5)
    segs = watershed(img, label(contours))
    segs, *_ = relabel_sequential(segs)
    return segs


def edgemap_segments_to_polygons(img,
                                 base_name='segment_{:d}',
                                 simplify_to_lines=False):
    segs = active_contour_segmentation(img)
    polys = polygonize(segs)
    if simplify_to_lines:
        yield from simplify_polys_to_lines(polys, segs)
    else:
        for i, (geom, _) in enumerate(polys):
            yield as_shape(geom), base_name.format(i), 0


def multilinestring_to_linestrings(mls):
    if mls.geom_type == 'LineString':
        return [mls, ]
    assert mls.is_simple
    linestring_coords = []
    for line_part in mls.geoms:
        xy1, xy2 = list(line_part.coords)
        for line in linestring_coords:
            if line[-1] == xy1:
                line.append(xy2)
                break
            elif line[-1] == xy2:
                line.append(xy1)
                break
            elif line[0] == xy1:
                line.insert(0, xy2)
                break
            elif line[0] == xy2:
                line.insert(0, xy1)
                break
        else:
            linestring_coords.append([xy1, xy2])
    linestrings = [LineString(coords) for coords in linestring_coords]
    return linestrings


def simplify_polys_to_lines(polys, segs, base_name='line_{:d}'):
    lines = []
    connections = RAG(segs)
    polys = {lab: as_shape(p) for p, lab in polys}
    for p1, p2 in connections.edges():
        intersections = polys[p1].intersection(polys[p2])
        if intersections.geom_type == 'GeometryCollection':
            for geom in intersections:
                if 'LineString' in geom.geom_type:
                    lines += multilinestring_to_linestrings(geom)
        elif 'LineString' in intersections.geom_type:
            lines += multilinestring_to_linestrings(intersections)
        else:
            assert False
    for i, line in enumerate(lines):
        yield line, base_name.format(i), 0


def heatmap_peaks_to_points(img, base_name='points_{:d}'):
    if img.ndim == 3:
        for z, z_img in enumerate(img, 1):
            for multipoint, *_ in heatmap_peaks_to_points(z_img):
                yield multipoint, base_name.format(z), z
    else:
        peaks = peak_local_max(img,
                               threshold_abs=0.5,
                               min_distance=5,
                               exclude_border=False,
                               indices=True)
        yield MultiPoint(peaks), base_name.format(0), 0


def _is_simple_rectangle(geom):
    minx, miny, maxx, maxy = geom.bounds
    xcoords, ycoords = np.asarray(geom.coords.xy)
    x_is_bounds = ((xcoords == minx) | (xcoords == maxx)).all()
    y_is_bounds = ((ycoords == miny) | (ycoords == maxy)).all()
    if len(geom.coords) == 5 and x_is_bounds and y_is_bounds:
        return True
    else:
        return False


def _geom_type_to_roi_type(geom):
    try:
        geom = geom.exterior
    except AttributeError:
        pass
    geom_type = geom.geom_type
    if geom_type == 'Point':
        return ('point',
                np.asarray([geom.x, ]),
                np.asarray([geom.y, ]))
    elif geom_type == 'MultiPoint':
        return ('point',
                np.asarray([g.x for g in geom]),
                np.asarray([g.y for g in geom]))
    elif geom_type == 'LinearRing':
        if _is_simple_rectangle(geom):
            return ('rect',
                    np.asarray(geom.coords.xy[0])[:-1],
                    np.asarray(geom.coords.xy[1])[:-1])
        else:
            return ('polygon',
                    np.asarray(geom.coords.xy[0])[:-1],
                    np.asarray(geom.coords.xy[1])[:-1])
    elif geom_type == 'LineString':
        if len(geom.coords.xy[0]) == 2:
            return ('line',
                    np.asarray(geom.coords.xy[0]),
                    np.asarray(geom.coords.xy[1]))
        else:
            return ('polyline',
                    np.asarray(geom.coords.xy[0]),
                    np.asarray(geom.coords.xy[1]))
    else:
        raise TypeError('geom_type is {}'.format(geom_type))


def geoms_to_roi_json(geoms):
    for g, name, z_position in geoms:
        roi = {
            'MAGIC': b'Iout',
            'VERSION_OFFSET': 226
        }
        roi_type, xcoords, ycoords = _geom_type_to_roi_type(g)
        # imagej is 1based
        xcoords = np.round(xcoords + 1).astype('i')
        ycoords = np.round(ycoords + 1).astype('i')
        roi['TYPE'] = ROI_TYPES[roi_type]
        if roi_type in ('rect', 'polygon', 'point', 'polyline'):
            roi['TOP'] = ycoords.min()
            roi['BOTTOM'] = ycoords.max()
            roi['LEFT'] = xcoords.min()
            roi['RIGHT'] = xcoords.max()
        elif roi_type == 'line':
            roi['X1'], roi['X2'] = xcoords
            roi['Y1'], roi['Y2'] = ycoords

        if roi_type in ('polygon', 'point', 'polyline'):
            roi['N_COORDINATES'] = len(xcoords)
            roi['COORDINATES'] = np.concatenate([xcoords - roi['LEFT'],
                                                 ycoords - roi['TOP']])
            roi['HEADER2_OFFSET'] = 64 + roi['N_COORDINATES'] * 4
        else:
            roi['HEADER2_OFFSET'] = 64
        roi['Z_POSITION'] = z_position
        roi['NAME'] = name
        yield roi


def write_geoms_as_rois(geoms, directory, zip_dir=True):
    if not os.path.exists(directory):
        os.mkdir(directory)
    elif not os.path.isdir(directory):
        raise IOError('{} is not a directory'.format(directory))
    for roi in geoms_to_roi_json(geoms):
        file_path = os.path.join(directory, roi['NAME'] + '.roi')
        with ROIEncoder(file_path) as r:
            r.write(roi)
    if zip_dir:
        z = zipfile.ZipFile(directory + '.zip', 'w', zipfile.ZIP_DEFLATED)
        for root, dirs, files in os.walk(directory):
            for fn in files:
                z.write(os.path.join(root, fn))
        z.close()
        shutil.rmtree(directory)
