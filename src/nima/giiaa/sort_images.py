"""
Using distribution-based GIIAA to make inference on a few random samples from the subset of images.
"""

import os
import random

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.nima.giiaa.base_module_giiaa import BaseModuleGIIAA


class LocalConfig(enumerate):
    GIIAA_WEIGHTS = 'models/giiaa-hist_200k_base-inceptionresnetv2_loss-0.078.hdf5'
    FOLDER_PATH = 'datasets/lab-groups/artificial5'


def get_weighted_mean(distribution):
    mean = 0.0
    for i in range(0, len(distribution)):
        mean += distribution[i] * (i + 1)
    return mean

def sort_images_by_mos(image_scores):
    # Sort the dictionary items by values (scores)
    sorted_images = dict(sorted(image_scores.items(), key=lambda item: item[1], reverse=False))
    return sorted_images

def run_model_multiple_images(folder_path):

    image_scores = {}
    for filename in os.listdir(folder_path):

        if not (filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png")):
            continue

        image_path = os.path.join(folder_path, filename)
        image = cv2.resize(cv2.imread(image_path), (224, 224)) / 255.0
        image = np.asarray(image)[np.newaxis, ...]

        prediction = nima.nima_model.predict(image)[0]
        image_scores[filename] = get_weighted_mean(prediction)

    return image_scores

if __name__ == "__main__":

    nima = BaseModuleGIIAA(custom_weights=LocalConfig.GIIAA_WEIGHTS)
    nima.build()
    nima.compile()

    image_scores = run_model_multiple_images(LocalConfig.FOLDER_PATH)
    image_scores = sort_images_by_mos(image_scores)

    print(LocalConfig.FOLDER_PATH)
    print(image_scores)





