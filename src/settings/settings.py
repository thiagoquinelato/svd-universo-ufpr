import os

PATHS = {
    'image_dir': 'images',
    'face_cascade_file': 'src/models/haarcascade_frontalface_default.xml',
    'eye_cascade_file': 'src/models/haarcascade_eye.xml',
    'left_eye_cascade_file': 'src/models/haarcascade_lefteye_2splits.xml',
    'right_eye_cascade_file': 'src/models/haarcascade_righteye_2splits.xml',
}

CAMERA = {
    'width': 640,
    'height': 480,
    'index': 0,
}

ACQUISITION = {
    'images_to_capture': 100,
    'n_threads': 4
}

FACE_DETECTION = {
    'scale_factor': 1.1,
    'min_neighbors': 5,
    'min_size': (30, 30),
    'highlight_color': (255, 0, 0),
    'apply_highlight': False,
    'max_rotation_angle': 20,  # Maximum allowed rotation angle in degrees
    'final_image_size': (128, 128)
}

EYE_DETECTION = {
    'scale_factor': 1.1,
    'min_neighbors': 5,
    'min_size': (10, 10),
    'highlight_color': (0, 255, 0),
    'apply_highlight': False,
}

RECOGNITION = {
    'n_threads': 4
}