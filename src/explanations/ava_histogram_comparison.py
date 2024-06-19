"""
Using the differently trained GCIAA models to make inference on a few random samples from the AVA test folder.
"""

import os
import random
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.nima.gciaa.base_module_gciaa import BaseModuleGCIAA
from src.musiq.siamese_transformer_base import SiameseTransformerBase

NIMA_MODEL = "models/nima_loss-0.078.hdf5"
ORIGINAL_IMAGES = "datasets/ava/images/test"

WEIGHTS = "models/koniq_ckpt.npz"
AVA_DATAFRAME_TEST_PATH = "datasets/ava/metadata/giiaa_metadata/dataframe_AVA_giiaa-hist_test.csv"

def get_mean(distribution):

    mean = 0.0
    for i in range(0, len(distribution)):
        mean += distribution[i] * (i + 1)
    return mean

if __name__ == "__main__":

    # Compile GCIAA model
    nima = BaseModuleGCIAA(custom_weights=NIMA_MODEL, load_weights_as='GIIAA')
    nima.build()
    nima.compile()

    musiq = SiameseTransformerBase(ckpt_path=WEIGHTS)

    dataframe = pd.read_csv(AVA_DATAFRAME_TEST_PATH, converters={'label': eval})
    
    predictions_nima__means = []
    predictions_musiq = []
    gt_means = []

    for i, orig_image_filename in enumerate(os.listdir(ORIGINAL_IMAGES)):

        # Get two random images and their groundtruth aesthetic value (histogram mean)
        orig_image_path = os.path.join(ORIGINAL_IMAGES, orig_image_filename)
        orig_image = cv2.resize(cv2.imread(orig_image_path), (224, 224)) / 255.0
        orig_image_np = np.asarray(orig_image)[np.newaxis, ...]

        try:
            print(f"Predicting {orig_image_path}")
            prediction_musiq = musiq.run_model_single_image(orig_image_path)
            print(f"Prediction: {prediction_musiq}")
            predictions_musiq.append(prediction_musiq / 10)
        except Exception as e:
            print(f"Error predicting {orig_image_path}: {e}")
            continue

        prediction_nima = nima.image_encoder_model.predict([orig_image_np])[0]
        gt = dataframe[dataframe["id"] == orig_image_path].iloc[0]['label']

        predictions_nima__means.append(get_mean(prediction_nima))
        gt_means.append(get_mean(gt))

        if i >= 640:
            break

    plt.hist(predictions_nima__means, bins=10, color='blue', edgecolor='black')
    plt.xlim(0, 10)
    plt.title('NIMA: Histogram of Prediction Mean Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

    plt.hist(predictions_musiq, bins=10, color='yellow', edgecolor='black')
    plt.xlim(0, 10)
    plt.title('MusiQ: Histogram of Prediction Mean Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

    plt.hist(gt_means, bins=10, color='green', edgecolor='black')
    plt.xlim(0, 10)
    plt.title('Histogram of Groundtruth Mean Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()