import os
import random
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.nima.gciaa.base_module_gciaa import BaseModuleGCIAA
from src.nima.giiaa.base_module_giiaa import BaseModuleGIIAA

"""
Evaluation of distortion-trained models for the master thesis version of IAA.

Task            |  Model            |   Accuracy
---------------------------------------------------
Black and White | BW-trained        | 85% (544/640)
Black and White | Base              | 23% (147/640)
Blur            | BL-trained        | 100% (640/640)
Blur            | Base              | 62% (399/640)
Blob Overlay    | BO-trained        | 97% (623/640)
Blob Overlay    | Base              | 62% (401/640)
Smiles-test     | Smiles-trained    | 70% (14/20)
Smiles-test     | Base              | 55% (11/20)
"""

GCIAA_MODEL = "models/nima_loss-0.078.hdf5"
# GCIAA_MODEL = "models/blob_overlay_0.956.hdf5"
# GCIAA_MODEL = "models/black_and_white_v2_0.939.hdf5"
# GCIAA_MODEL = "models/smiles_0.982.hdf5"
# GCIAA_MODEL = "models/blur_0.995.hdf5"


ORIGINAL_IMAGES = "datasets/distortions/variants_v2/original"
DISTORTED_IMAGES = "datasets/distortions/variants_v2/blur_v2"

# ORIGINAL_IMAGES = "datasets/portraits/smile_real"
# DISTORTED_IMAGES = "datasets/portraits/frown_real"


def find_matching_image_path(filename, folder1=ORIGINAL_IMAGES, folder2=DISTORTED_IMAGES):
    index = filename.split('_')[1]

    for file in os.listdir(folder2):
        if file.startswith('_' + index):
            return os.path.join(folder2, file)

def evaluate_distortions():
    gciaa = BaseModuleGCIAA(custom_weights=GCIAA_MODEL, load_weights_as='GIIAA')
    gciaa.build()
    gciaa.compile()

    count_all = 0
    count_correct = 0

    orig_image_filenames = [name for name in os.listdir(ORIGINAL_IMAGES) if not name.startswith('.')]
    distorted_image_filenames = [name for name in os.listdir(DISTORTED_IMAGES) if not name.startswith('.')]

    assert len(orig_image_filenames) == len(distorted_image_filenames)

    predictions = []

    for i in range(len(orig_image_filenames)):

        orig_image_filename = orig_image_filenames[i]
        orig_image_path = os.path.join(ORIGINAL_IMAGES, orig_image_filename)
        orig_image = cv2.resize(cv2.imread(orig_image_path), (224, 224)) / 255.0
        orig_image_np = np.asarray(orig_image)[np.newaxis, ...]

        # distorted_image_path = find_matching_image_path(orig_image_filename)
        distorted_image_filename = distorted_image_filenames[i]
        distorted_image_path = os.path.join(DISTORTED_IMAGES, distorted_image_filename)
        distorted_image = cv2.resize(cv2.imread(distorted_image_path), (224, 224)) / 255
        distorted_image_np = np.asarray(distorted_image)[np.newaxis, ...]

        # Make inference with GCIAA BO model
        gciaa_prediction = gciaa.predict([orig_image_np, distorted_image_np])[0, 0]
        print(gciaa_prediction)

        predictions.append(gciaa_prediction)

        count_all += 1
        if gciaa_prediction < 0.5:
            count_correct += 1


    print(f"Count correct: {count_correct}")
    print(f"Count all: {count_all}")
    print(f"Accuracy: {count_correct / count_all}")

    plt.hist(predictions, bins=20, edgecolor='black')
    plt.title('Histogram of Predictions')
    plt.xlabel('Prediction Value')
    plt.ylabel('Frequency')
    plt.xlim(0, 1)
    plt.show()


if __name__ == "__main__":
    evaluate_distortions()
    
