import os
import cv2
import numpy as np
import h5py
from sklearn.utils import shuffle

from src.preprocessing import preprocess_detection_image, preprocess_mat_image



# 1. Detection Dataset (RGB)

def load_detection_dataset(dataset_path):

    data = []
    labels = []

    categories = ["no", "yes"]

    for category in categories:

        path = os.path.join(dataset_path, category)
        label = categories.index(category)

        for img_name in os.listdir(path):

            img_path = os.path.join(path, img_name)

            img = cv2.imread(img_path)

            if img is None:
                continue

            # Convert BGR → RGB (IMPORTANT for pretrained models)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Preprocess (resize + normalize)
            img = preprocess_detection_image(img)

            data.append(img)
            labels.append(label)

    X = np.array(data)
    y = np.array(labels)

    # Shuffle dataset
    X, y = shuffle(X, y, random_state=42)

    # Debug check (VERY IMPORTANT)
    print("Detection Dataset Shape:", X.shape)

    return X, y



# 2. Figshare Dataset (Grayscale)

def load_figshare_dataset(dataset_path):

    X = []
    y = []

    files = os.listdir(dataset_path)

    for file in files:

        if file.endswith(".mat"):

            file_path = os.path.join(dataset_path, file)

            with h5py.File(file_path, 'r') as data:

                image = np.array(data['cjdata']['image']).T
                label = int(np.array(data['cjdata']['label'])[0][0])

                # Convert labels from (1,2,3) → (0,1,2)
                label = label - 1

                image = preprocess_mat_image(image)

                X.append(image)
                y.append(label)

    X = np.array(X)
    y = np.array(y)

    # Shuffle dataset
    X, y = shuffle(X, y, random_state=42)

    # Debug check
    print("Classification Dataset Shape:", X.shape)

    return X, y


# debugg dataset_loader
# if __name__ == "__main__":
    
#     X_det, y_det = load_detection_dataset("data/tumor_detection_dataset")
#     print("Detection shape:", X_det.shape)

#     X_type, y_type = load_figshare_dataset("data/figshare_dataset/brainTumorDataPublic_1-766")
#     print("Classification shape:", X_type.shape)