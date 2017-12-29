import os
import struct
import zipfile
import shutil
import numpy as np

HEADER1_FIELDS = {
    # 'VAR_NAME', 'type', offset'
    'MAGIC': ['4s', 0],
    'VERSION_OFFSET': ['h', 4],
    'TYPE': ['b', 6],
    'TOP': ['h', 8],
    'LEFT': ['h', 10],
    'BOTTOM': ['h', 12],
    'RIGHT': ['h', 14],
    'N_COORDINATES': ['h', 16],
    'X1': ['f', 18],
    'Y1': ['f', 22],
    'X2': ['f', 26],
    'Y2': ['f', 30],
    'XD': ['f', 18],  # D vars for sub pixel resolution ROIs
    'YD': ['f', 22],
    'WIDTH': ['f', 26],
    'HEIGHT': ['f', 30],
    'STROKE_WIDTH': ['h', 34],
    'SHAPE_ROI_SIZE': ['i', 36],
    'STROKE_COLOR': ['i', 40],
    'FILL_COLOR': ['i', 44],
    'SUBTYPE': ['h', 48],
    'OPTIONS': ['h', 50],
    'ARROW_STYLE': ['b', 52],
    'ELLIPSE_ASPECT_RATIO': ['b', 52],
    'POINT_TYPE': ['b', 52],
    'ARROW_HEAD_SIZE': ['b', 53],
    'ROUNDED_RECT_ARC_SIZE': ['h', 54],
    'POSITION': ['i', 56],
    'HEADER2_OFFSET': ['i', 60],
}


HEADER2_FIELDS = {
    'C_POSITION': ['i', 4],
    'Z_POSITION': ['i', 8],
    'T_POSITION': ['i', 12],
    'NAME_OFFSET': ['i', 16],
    'NAME_LENGTH': ['i', 20],
    'OVERLAY_LABEL_COLOR': ['i', 24],
    'OVERLAY_FONT_SIZE': ['h', 28],
    'AVAILABLE_BYTE1': ['b', 30],
    'IMAGE_OPACITY': ['b', 31],
    'IMAGE_SIZE': ['i', 32],
    'FLOAT_STROKE_WIDTH': ['f', 36],
    'ROI_PROPS_OFFSET': ['i', 40],
    'ROI_PROPS_LENGTH': ['i', 44]
}


ROI_TYPES = {
    'polygon': 0, 'rect': 1, 'oval': 2,
    'line': 3, 'freeline': 4,
    'polyline': 5, 'no_roi': 6,
    'freehand': 7, 'traced': 8,
    'angle': 9, 'point': 10
}


class ROIEncoder(object):

    def __init__(self, file_path):
        self.path = file_path
        self._file = None
        self.is_open = False

    def open(self):
        self._file = open(self.path, 'wb')
        pad = struct.pack('128b', *np.zeros(128, dtype='i'))
        self._file.write(pad)
        self.is_open = True
        return self

    def close(self):
        self._file.close()
        self.is_open = False

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def write(self, roi_json):
        for var_name, value in roi_json.items():
            if var_name == 'COORDINATES':
                self.write_coords(value)
            elif var_name == 'NAME':
                self.write_name(value, roi_json['HEADER2_OFFSET'])
            else:
                try:
                    struct_type, offset = HEADER1_FIELDS[var_name]
                except KeyError:
                    struct_type, offset = HEADER2_FIELDS[var_name]
                    offset += roi_json['HEADER2_OFFSET']
                self.write_var(struct_type, offset, value)

    def write_var(self, struct_type, offset, value):
        self._file.seek(offset)
        self._file.write(struct.pack('>' + struct_type, value))

    def write_coords(self, coords):
        self._file.seek(64)
        self._file.write(struct.pack('>{:d}h'.format(len(coords)), *coords))

    def write_name(self, name, h2_offset):
        # NAME OFFSET
        name_offset = h2_offset + 64
        self.write_var(
            HEADER2_FIELDS['NAME_OFFSET'][0],
            HEADER2_FIELDS['NAME_OFFSET'][1] + h2_offset,
            name_offset
        )

        # NAME LENGTH
        self.write_var(
            HEADER2_FIELDS['NAME_LENGTH'][0],
            HEADER2_FIELDS['NAME_LENGTH'][1] + h2_offset,
            len(name)
        )
        self._file.seek(name_offset)
        self._file.write(name.encode())


def is_simple_rectangle(geom):
    minx, miny, maxx, maxy = geom.bounds
    xcoords, ycoords = np.asarray(geom.coords.xy)
    x_is_bounds = ((xcoords == minx) | (xcoords == maxx)).all()
    y_is_bounds = ((ycoords == miny) | (ycoords == maxy)).all()
    if len(geom.coords) == 5 and x_is_bounds and y_is_bounds:
        return True
    else:
        return False


def geom_type_to_roi_type(geom):
    try:
        geom = geom.exterior
    except AttributeError:
        pass
    geom_type = geom.geom_type
    if geom_type == 'Point':
        return ('point',
                np.asarray([geom.x, ], dtype='i'),
                np.asarray([geom.y, ], dtype='i'))
    elif geom_type == 'MultiPoint':
        return ('point',
                np.asarray([g.x for g in geom], dtype='i'),
                np.asarray([g.y for g in geom], dtype='i'))
    elif geom_type == 'LinearRing':
        if is_simple_rectangle(geom):
            return ('rect',
                    np.asarray(geom.coords.xy[0], dtype='i')[:-1],
                    np.asarray(geom.coords.xy[1], dtype='i')[:-1])
        else:
            return ('polygon',
                    np.asarray(geom.coords.xy[0], dtype='i')[:-1],
                    np.asarray(geom.coords.xy[1], dtype='i')[:-1])
    elif geom_type == 'LineString':
        if len(geom.coords.xy[0]) == 2:
            return ('line',
                    np.asarray(geom.coords.xy[0], dtype='i'),
                    np.asarray(geom.coords.xy[1], dtype='i'))
        else:
            return ('polyline',
                    np.asarray(geom.coords.xy[0], dtype='i'),
                    np.asarray(geom.coords.xy[1], dtype='i'))


def geoms_to_roi_json(geoms):
    for g, name, z_position in geoms:
        roi = {
            'MAGIC': b'Iout',
            'VERSION_OFFSET': 226
        }
        roi_type, xcoords, ycoords = geom_type_to_roi_type(g)
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
        roi['C_POSITION'] = z_position
        roi['T_POSITION'] = z_position
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
