import cv2
import numpy as np

IMG_SIZE = 224



# 1. Detection Dataset (RGB)

def preprocess_detection_image(img):

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # convert to RGB (3 channels)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


# 2. Figshare MRI Dataset (.mat)

def preprocess_mat_image(image):

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    # Normalize
    image = image / 255.0

    # Add channel dimension
    image = np.expand_dims(image, axis=-1)

    return image