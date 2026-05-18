import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk

from typing import Optional

import cv2
import os
import logging
import numpy as np

import matplotlib.pyplot as plt

from settings.settings import PATHS, CAMERA, FACE_DETECTION, EYE_DETECTION


# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_image_dir():
    return os.path.join(os.getcwd(), PATHS['image_dir'])

def get_face_cascade_path():
    return os.path.join(os.getcwd(), PATHS['face_cascade_file'])

face_cascade = None
def get_face_cascade():
    global face_cascade
    if face_cascade is None:
        face_cascade_path = get_face_cascade_path()
        if not os.path.exists(face_cascade_path):
            logger.error(f"Face cascade file not found: {face_cascade_path}")
            raise FileNotFoundError(f"Face cascade file not found: {face_cascade_path}")
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        if face_cascade.empty():
            logger.error("Error loading cascade classifier")
            raise ValueError("Error loading cascade classifier")
    return face_cascade

def get_eye_cascade_path():
    return os.path.join(os.getcwd(), PATHS['eye_cascade_file'])

eye_cascade = None
def get_eye_cascade():
    global eye_cascade
    if eye_cascade is None:
        eye_cascade_path = get_eye_cascade_path()
        if not os.path.exists(eye_cascade_path):
            logger.error(f"Eye cascade file not found: {eye_cascade_path}")
            raise FileNotFoundError(f"Eye cascade file not found: {eye_cascade_path}")
        eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        if eye_cascade.empty():
            logger.error("Error loading eye cascade classifier")
            raise ValueError("Error loading eye cascade classifier")
    return eye_cascade

left_eye_cascade = None
def get_left_eye_cascade():
    global left_eye_cascade
    if left_eye_cascade is None:
        left_eye_cascade_path = os.path.join(os.getcwd(), PATHS['left_eye_cascade_file'])
        if not os.path.exists(left_eye_cascade_path):
            logger.error(f"Left eye cascade file not found: {left_eye_cascade_path}")
            raise FileNotFoundError(f"Left eye cascade file not found: {left_eye_cascade_path}")
        left_eye_cascade = cv2.CascadeClassifier(left_eye_cascade_path)
        if left_eye_cascade.empty():
            logger.error("Error loading left eye cascade classifier")
            raise ValueError("Error loading left eye cascade classifier")
    return left_eye_cascade

right_eye_cascade = None
def get_right_eye_cascade():
    global right_eye_cascade
    if right_eye_cascade is None:
        right_eye_cascade_path = os.path.join(os.getcwd(), PATHS['right_eye_cascade_file'])
        if not os.path.exists(right_eye_cascade_path):
            logger.error(f"Right eye cascade file not found: {right_eye_cascade_path}")
            raise FileNotFoundError(f"Right eye cascade file not found: {right_eye_cascade_path}")
        right_eye_cascade = cv2.CascadeClassifier(right_eye_cascade_path)
        if right_eye_cascade.empty():
            logger.error("Error loading right eye cascade classifier")
            raise ValueError("Error loading right eye cascade classifier")
    return right_eye_cascade


def get_camera_index():
    return CAMERA['index']

def create_directory(directory: str) -> None:
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
    except OSError as e:
        logger.error(f"Error creating directory {directory}: {e}")
        raise

def initialize_camera(camera_index: int = 0) -> Optional[cv2.VideoCapture]:
    """
    Initialize the camera with error handling
    
    Parameters:
        camera_index (int): Camera device index
    Returns:
        cv2.VideoCapture or None: Initialized camera object
    """
    try:
        cam = cv2.VideoCapture(camera_index)
        if not cam.isOpened():
            logger.error("Could not open webcam")
            return None
            
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA['width'])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA['height'])
        return cam
    except Exception as e:
        logger.error(f"Error initializing camera: {e}")
        return None

def detect_face_and_eyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = get_face_cascade().detectMultiScale(
        gray,
        scaleFactor=FACE_DETECTION['scale_factor'],
        minNeighbors=FACE_DETECTION['min_neighbors'],
        minSize=FACE_DETECTION['min_size']
    )

    # keep only the largest face detected
    if len(faces) == 0:
        return None, None, None, None
    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
    face_x, face_y, face_w, face_h = faces[0]
    
    if FACE_DETECTION['apply_highlight']:
        cv2.rectangle(frame, (face_x, face_y), (face_x+face_w, face_y+face_h), FACE_DETECTION['highlight_color'], 2)

    face_img = frame[face_y:face_y+face_h, face_x:face_x+face_w]
    face_img_gray = gray[face_y:face_y+face_h, face_x:face_x+face_w]

    left_eye_cascade = get_left_eye_cascade()
    left_eyes = left_eye_cascade.detectMultiScale(face_img_gray, scaleFactor=EYE_DETECTION['scale_factor'], minNeighbors=EYE_DETECTION['min_neighbors'], minSize=EYE_DETECTION['min_size'])
    right_eye_cascade = get_right_eye_cascade()
    right_eyes = right_eye_cascade.detectMultiScale(face_img_gray, scaleFactor=EYE_DETECTION['scale_factor'], minNeighbors=EYE_DETECTION['min_neighbors'], minSize=EYE_DETECTION['min_size'])
    if len(left_eyes) == 0:
        eyes = right_eyes
    elif len(right_eyes) == 0:
        eyes = left_eyes
    else:
        eyes = np.concatenate([left_eyes, right_eyes])        
    if len(eyes) < 2:
        return None, None, None, None

    eyes = sorted(eyes, key=lambda x: x[0])
    eyes = [eyes[0], eyes[-1]]
    rotation_center = [0, 0]
    for (eye_x, eye_y, eye_w, eye_h) in eyes:  
        eye_center = (eye_x + eye_w // 2, eye_y + eye_h // 2)
        rotation_center = (rotation_center[0] + eye_center[0], rotation_center[1] + eye_center[1])
        if EYE_DETECTION['apply_highlight']:
            cv2.circle(face_img, eye_center, 5, EYE_DETECTION['highlight_color'], -1)
    rotation_center = (int(face_x + rotation_center[0] // 2), int(face_y + rotation_center[1] // 2))
    logger.info(f"rotation center (global coordinates): {rotation_center}")
    dx = eyes[1][0] - eyes[0][0]
    dy = eyes[1][1] - eyes[0][1]
    angle = np.degrees(np.arctan2(dy, dx))
    logger.info(f"rotation angle: {angle:.2f} degrees")
    M = cv2.getRotationMatrix2D(center=rotation_center, angle=angle, scale=1)
    logger.info(f"Rotation matrix: {M}")
    frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
    gray = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]))
    face_img = frame[face_y:face_y+face_h, face_x:face_x+face_w]
    face_img_gray = gray[face_y:face_y+face_h, face_x:face_x+face_w]

    return frame, gray, face_img, face_img_gray

capturing = False
capturedFrame = None

def detect():
    global capturedFrame
    frame = capturedFrame.copy() 
    capturedFrame = None

    frame, gray, face_img, face_img_gray = detect_face_and_eyes(frame)

    global captureStep
    if frame is None:
        captureStep = 6
    else: 
        captureStep = 7

    # if frame is not None:
    #     img_path = f'./{PATHS["image_dir"]}/img.jpg'
    #     cv2.imwrite(img_path, frame)

    # if gray is not None:
    #     img_path = f'./{PATHS["image_dir"]}/img_gray.jpg'
    #     cv2.imwrite(img_path, gray)

    # if face_img is not None:
    #     img_path = f'./{PATHS["image_dir"]}/face_img.jpg'
    #     cv2.imwrite(img_path, face_img)

    if face_img_gray is not None:
        img_path = f'./{PATHS["image_dir"]}/face_img_gray.jpg'
        cv2.imwrite(img_path, face_img_gray)


def capture():
    global capturing, captureStep
    capturing = True
    captureStep = 0
    btnCapturar.config(state=DISABLED)
    root.after(1000, increase_capture_step)
    root.after(2000, increase_capture_step)
    root.after(3000, increase_capture_step)
    root.after(3500, increase_capture_step)


def increase_capture_step():
    global captureStep
    captureStep += 1

captureStep = 0
def camera_loop():
    ret, frame = cam.read()
    if not ret:
        logger.warning("Failed to grab frame")
        # repetir a tentativa de captura

    global capturing
    if capturing:

        text = ""
        global captureStep
        if captureStep == 0:
            text = "3"
        elif captureStep == 1:
            text = "2"
        elif captureStep == 2:
            text = "1"
        elif captureStep == 3:
            text = "Capturando..."
        elif captureStep == 4:
            global capturedFrame
            capturedFrame = frame.copy()
            btnCapturar.after(10, detect)
            captureStep += 1
            text = "Aguarde..."
        elif captureStep == 5:
            text = "Aguarde..."
        elif captureStep == 6:
            text = "Tente novamente."
            btnCapturar.config(state=NORMAL)
        elif captureStep == 7:
            capturing = False
            captureStep = 0
            btnCapturar.config(state=NORMAL)

        if captureStep < 7:
            # Define text properties
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            color = (200, 200, 200)  # BGR format 
            thickness = 3
            text_width, text_height = cv2.getTextSize(text, font, font_scale, thickness)[0]
            top_center_coordinates = (int(frame.shape[1] / 2) - int(text_width / 2), 2*text_height)
            cv2.putText(frame, text, top_center_coordinates, font, font_scale, color, thickness, cv2.LINE_AA)

    photo_image = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)))
    label_camera.photo_image = photo_image
    label_camera.configure(image=photo_image)
    label_camera.after(10, camera_loop)

if __name__ == '__main__':
    try:
        # Initialize
        create_directory(get_image_dir())
            
        cam = initialize_camera(get_camera_index())
        if cam is None:
            raise ValueError("Failed to initialize camera")
                        
        logger.info(f"Initializing face capture")
        logger.info("Look at the camera and wait...")
        
        root = tk.Tk()
        root.bind('<Escape>', lambda e: root.quit())

        root.title("Universo UFPR")

        label_camera = ttk.Label(root)
        label_camera.pack(side=LEFT, padx=10, pady=10)

        btnCapturar = ttk.Button(root, text="Capturar", command=capture, bootstyle="success")
        btnCapturar.pack(side=LEFT, padx=5, pady=10)

        label_famous = ttk.Label(root, width=80)
        label_famous.pack(side=LEFT, padx=10, pady=10)

        camera_loop()

        root.mainloop()
        #camera_loop(cam)

        # while True:
        #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #     faces = face_cascade.detectMultiScale(
        #         gray,
        #         scaleFactor=FACE_DETECTION['scale_factor'],
        #         minNeighbors=FACE_DETECTION['min_neighbors'],
        #         minSize=FACE_DETECTION['min_size']
        #     )
            
        #     for (x, y, w, h) in faces:
        #         cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
        #         face_img = img[y:y+h, x:x+w]
        #         face_img_gray = gray[y:y+h, x:x+w]
        #         img_path = f'./{PATHS["image_dir"]}/img.jpg'
        #         cv2.imwrite(img_path, img)
        #         img_path = f'./{PATHS["image_dir"]}/face_img_gray.jpg'
        #         cv2.imwrite(img_path, face_img_gray)
        #         img_path = f'./{PATHS["image_dir"]}/face_img.jpg'
        #         cv2.imwrite(img_path, face_img)
                
        #         image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #         # Plota com matplotlib
        #         # plt.imshow(image_rgb)
        #         # plt.axis('off')
        #         # plt.show()
            
        #     #cv2.imshow('Face Capture', img)
            
        #     if cv2.waitKey(100) & 0xff == 27:  # ESC key
        #         break
                
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        
    finally:
        if 'cam' in locals():
            cam.release()
        cv2.destroyAllWindows()