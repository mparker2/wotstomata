import struct
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
