import tensorflow as tf
import random
import numpy as np
import os
import cv2

MODEL_PATH = "models/brightness_predictorinceptionresnetv2_0.008.hdf5"
IMAGES_PATH = "datasets/brightness-artificial/images"

if __name__ == "__main__":
    model = tf.keras.models.load_model(MODEL_PATH)

    for filename in os.listdir(IMAGES_PATH):
        
        if not filename.endswith(".jpg"):
            continue

        file = os.path.join(IMAGES_PATH, filename)

        image = cv2.resize(cv2.imread(file), (224, 224)) / 255.0
        image = np.asarray(image)[np.newaxis, ...]

        outputs = model.predict(image)
        print("{}: {}".format(file, outputs))