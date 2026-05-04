from tkinter import *
from PIL import Image, ImageTk

from typing import Optional

import cv2
import os
import logging

import matplotlib.pyplot as plt

from settings.settings import PATHS, CAMERA, FACE_DETECTION


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_image_dir():
    return os.path.join(os.getcwd(), PATHS['image_dir'])

def get_face_cascade_path():
    return os.path.join(os.getcwd(), PATHS['face_cascade_file'])

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
    
def camera_loop():
    ret, frame = cam.read()
    if not ret:
        logger.warning("Failed to grab frame")
        # repetir a tentativa de captura

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=FACE_DETECTION['scale_factor'],
        minNeighbors=FACE_DETECTION['min_neighbors'],
        minSize=FACE_DETECTION['min_size']
    )
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        face_img = frame[y:y+h, x:x+w]
        face_img_gray = gray[y:y+h, x:x+w]
        img_path = f'./{PATHS["image_dir"]}/img.jpg'
        cv2.imwrite(img_path, frame)
        img_path = f'./{PATHS["image_dir"]}/face_img_gray.jpg'
        cv2.imwrite(img_path, face_img_gray)
        img_path = f'./{PATHS["image_dir"]}/face_img.jpg'
        cv2.imwrite(img_path, face_img)

    opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

    captured_image = Image.fromarray(opencv_image)

    photo_image = ImageTk.PhotoImage(image=captured_image)
    label_widget.photo_image = photo_image
    label_widget.configure(image=photo_image)
    label_widget.after(10, camera_loop)

if __name__ == '__main__':
    try:
        # Initialize
        create_directory(get_image_dir())
        face_cascade = cv2.CascadeClassifier(get_face_cascade_path())
        if face_cascade.empty():
            raise ValueError("Error loading cascade classifier")
            
        cam = initialize_camera(get_camera_index())
        if cam is None:
            raise ValueError("Failed to initialize camera")
                        
        logger.info(f"Initializing face capture")
        logger.info("Look at the camera and wait...")
        
        app = Tk()
        app.bind('<Escape>', lambda e: app.quit())

        app.title("Universo UFPR")

        label_widget = Label(app)
        label_widget.pack()

        camera_loop()

        app.mainloop()
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