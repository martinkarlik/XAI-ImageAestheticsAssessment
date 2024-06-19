"""
Using distribution-based GIIAA to make inference on a few random samples from the subset of images.
"""

import os
import random

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.nima.giiaa.base_module_giiaa import BaseModuleGIIAA


class LocalConfig(enumerate):
    GIIAA_WEIGHTS = 'models/nima_loss-0.078.hdf5'
    FOLDER_PATH = 'docs/NIMA-outputs/NIMA-histograms'
    SPECIFIC_FILE = 'datasets/ava/images/test/4943.jpg'
    OUTPUT_PATH = 'docs/NIMA-outputs/NIMA-histograms'


def get_weighted_mean(distribution):
    mean = 0.0
    for i in range(0, len(distribution)):
        mean += distribution[i] * (i + 1)
    return mean

def get_mode(distribution):
    return distribution.index(max(distribution)) + 1

def build_histogram(data, image_name):
    
    x_values = range(1, 11)

    plt.figure(figsize=(10, 5))
    plt.bar(x_values, data, color='skyblue', edgecolor='black')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.title(image_name)
    plt.savefig(os.path.join(LocalConfig.OUTPUT_PATH, image_name + '_histogram.png'))

def save_histograms_from_images(folder_path=None, specific_file=None):

    if specific_file:
        image = cv2.resize(cv2.imread(specific_file), (224, 224)) / 255.0
        image = np.asarray(image)[np.newaxis, ...]

        prediction = nima.nima_model.predict(image)[0]
        build_histogram(prediction, specific_file.split('/')[-1])
        return

    if not folder_path:
        return

    for filename in os.listdir(folder_path):

        if not (filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png")):
            continue

        image_path = os.path.join(folder_path, filename)
        image = cv2.resize(cv2.imread(image_path), (224, 224)) / 255.0
        image = np.asarray(image)[np.newaxis, ...]

        prediction = nima.nima_model.predict(image)[0]
        build_histogram(prediction, filename)

if __name__ == "__main__":

    nima = BaseModuleGIIAA(custom_weights=LocalConfig.GIIAA_WEIGHTS)
    nima.build()
    nima.compile()

    save_histograms_from_images(specific_file=LocalConfig.SPECIFIC_FILE)

    print(LocalConfig.FOLDER_PATH)




    