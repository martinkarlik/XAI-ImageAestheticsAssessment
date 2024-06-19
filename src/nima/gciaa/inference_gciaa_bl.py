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
from src.nima.giiaa.base_module_giiaa import BaseModuleGIIAA

GCIAA_NIMA_MODEL = "models/nima_loss-0.078.hdf5"
GCIAA_BL_MODEL = "models/blur_0.995.hdf5"

ORIGINAL_IMAGES = "datasets/distortions/variants_v2/original"
BL_IMAGES = "datasets/distortions/variants_v2/blur_v2"

def get_mean(distribution):

    mean = 0.0
    for i in range(0, len(distribution)):
        mean += distribution[i] * (i + 1)
    return mean

def find_matching_image_path(filename, folder1=ORIGINAL_IMAGES, folder2=BL_IMAGES):
    index = filename.split('_')[1]

    for file in os.listdir(folder2):
        if file.startswith('_' + index):
            return os.path.join(folder2, file)

    return None

if __name__ == "__main__":

    # Compile GCIAA model
    gciaa = BaseModuleGCIAA(custom_weights=GCIAA_BL_MODEL, load_weights_as='GCIAA')
    gciaa.build()
    gciaa.compile()

    gciaa_nima = BaseModuleGCIAA(custom_weights=GCIAA_NIMA_MODEL, load_weights_as='GIIAA')
    gciaa_nima.build()
    gciaa_nima.compile()

    for i in range(10):

        orig_image_filename = random.choice(os.listdir(ORIGINAL_IMAGES))
        # Get two random images and their groundtruth aesthetic value (histogram mean)
        orig_image_path = os.path.join(ORIGINAL_IMAGES, orig_image_filename)
        orig_image = cv2.resize(cv2.imread(orig_image_path), (224, 224)) / 255.0
        orig_image_np = np.asarray(orig_image)[np.newaxis, ...]

        bl_image_path = find_matching_image_path(orig_image_filename)
        bl_image = cv2.resize(cv2.imread(bl_image_path), (224, 224)) / 255
        bl_image_np = np.asarray(bl_image)[np.newaxis, ...]

        # Make inference with GCIAA BL model
        gciaa_prediction = gciaa.predict([orig_image_np, bl_image_np])[0, 0]
        print("Comparison BL: {:.2f}".format(gciaa_prediction))

        orig_image_prediction = gciaa.predict_image_encoder(orig_image_np)[0]
        bl_image_prediction = gciaa.predict_image_encoder(bl_image_np)[0]

        indices = np.arange(len(orig_image_prediction))
        width = 0.35
        plt.figure(figsize=(10, 6))
        plt.bar(indices, orig_image_prediction, width, label='Original Image Prediction', alpha=0.7)
        plt.bar(indices + width, bl_image_prediction, width, label='Blurry Image Prediction', alpha=0.7)

        cv2.imshow('Original Image', orig_image)
        cv2.imshow('Blurry Image', bl_image)

        plt.title('Distribution of Predicted Scores using Blur-trained model')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')
        plt.show()


        # Make inference with GCIAA Base model
        gciaa_prediction = gciaa_nima.predict([orig_image_np, bl_image_np])[0, 0]
        print("Comparison Base: {:.2f}".format(gciaa_prediction))

        orig_image_prediction = gciaa_nima.predict_image_encoder(orig_image_np)[0]
        bl_image_prediction = gciaa_nima.predict_image_encoder(bl_image_np)[0]

        indices = np.arange(len(orig_image_prediction))
        width = 0.35
        plt.figure(figsize=(10, 6))
        plt.bar(indices, orig_image_prediction, width, label='Original Image Prediction', alpha=0.7)
        plt.bar(indices + width, bl_image_prediction, width, label='Blurry Image Prediction', alpha=0.7)

        cv2.imshow('Original Image', orig_image)
        cv2.imshow('Blurry Image', bl_image)

        plt.title('Distribution of Predicted Scores using base model')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')
        plt.show()