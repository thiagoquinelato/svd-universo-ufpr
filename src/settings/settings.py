import os

PATHS = {
    'image_dir': 'images',
    'face_cascade_file': 'models/haarcascade_frontalface_default.xml',
}

CAMERA = {
    'width': 640,
    'height': 480,
    'index': 0,
}

FACE_DETECTION = {
    'scale_factor': 1.1,
    'min_neighbors': 5,
    'min_size': (30, 30),
}