"""
Evaluation of the GIIAA model.
Evaluated on 51 100 images from AVA dataset.

Earth mover's distance loss: 0.0772
"""

import sys
import pandas as pd
import os
import numpy as np
import cv2
from tqdm import tqdm
from absl import flags

from src.nima.giiaa.base_module_giiaa import BaseModuleGIIAA


GIIAA_PATH = "models/giiaa-hist_200k_base-inceptionresnetv2_loss-0.078.hdf5"
AVA_DATASET_TEST_PATH = "datasets/ava/images/test"
AVA_DATAFRAME_TEST_PATH = "datasets/ava/metadata/giiaa_metadata/dataframe_AVA_giiaa-hist_test.csv"
BATCH_SIZE = 1

TEST_SIZE = 1000

def get_weighted_mean(distribution):
    mean = 0.0
    for i in range(0, len(distribution)):
        mean += distribution[i] * (i + 1)
    return mean

def evaluate_nima(dataset_path):
    nima = BaseModuleGIIAA()
    nima.build()
    nima.compile()

    dataframe = pd.read_csv(AVA_DATAFRAME_TEST_PATH, converters={'label': eval})

    count_all = 0
    count_correct = 0

    for image_filename in tqdm(os.listdir(dataset_path)):
        image_path = os.path.join(dataset_path, image_filename)
        image = cv2.resize(cv2.imread(image_path), (224, 224)) / 255.0
        image = np.asarray(image)[np.newaxis, ...]

        gt = dataframe[dataframe["id"] == image_path].iloc[0]['label']
        prediction = nima.nima_model.predict(image)[0]

        prediction_mean = get_weighted_mean(prediction)
        gt_mean = get_weighted_mean(gt)
        
        count_correct += 1 if prediction_mean > 5.5 and gt_mean > 5.5 or prediction_mean < 5.5 and gt_mean < 5.5 else 0
        count_all += 1

        if count_all == TEST_SIZE:
            break

    print(f"Count correct: {count_correct}")
    print(f"Count all: {count_all}")
    print(f"Accuracy: {count_correct / count_all}")


if __name__ == "__main__":
    evaluate_nima(AVA_DATASET_TEST_PATH)
    
