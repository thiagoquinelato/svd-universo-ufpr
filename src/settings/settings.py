import os

PATHS = {
    'image_dir': 'images',
    'face_cascade_file': 'src/models/haarcascade_frontalface_default.xml',
    'eye_cascade_file': 'src/models/haarcascade_eye.xml',
    'left_eye_cascade_file': 'src/models/haarcascade_lefteye_2splits.xml',
    'right_eye_cascade_file': 'src/models/haarcascade_righteye_2splits.xml',
}

CAMERA = {
    'width': 800,
    'height': 600,
    'index': 0,
}

FACE_DETECTION = {
    'scale_factor': 1.1,
    'min_neighbors': 5,
    'min_size': (30, 30),
    'highlight_color': (255, 0, 0),
    'apply_highlight': False,
}

EYE_DETECTION = {
    'scale_factor': 1.1,
    'min_neighbors': 5,
    'min_size': (10, 10),
    'highlight_color': (0, 255, 0),
    'apply_highlight': False,
}