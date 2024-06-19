

import sys
import pandas as pd
import os
import numpy as np
import cv2
from tqdm import tqdm
from absl import flags

from src.musiq.siamese_transformer_base import SiameseTransformerBase


AVA_DATASET_TEST_PATH = "datasets/ava/images/test"
AVA_DATAFRAME_TEST_PATH = "datasets/ava/metadata/giiaa_metadata/dataframe_AVA_giiaa-hist_test.csv"
BATCH_SIZE = 1

MUSIQ_MEAN = 63.94
GT_MEAN = 5.40

TEST_SIZE = 100

def get_weighted_mean(distribution):
    mean = 0.0
    for i in range(0, len(distribution)):
        mean += distribution[i] * (i + 1)
    return mean

def evaluate_musiq_high_low(dataset_path):
    musiq = SiameseTransformerBase()

    dataframe = pd.read_csv(AVA_DATAFRAME_TEST_PATH, converters={'label': eval})

    count_all = 0
    count_correct = 0

    for image_filename in tqdm(os.listdir(dataset_path)):
        image_path = os.path.join(dataset_path, image_filename)
        image = cv2.resize(cv2.imread(image_path), (224, 224)) / 255.0
        image = np.asarray(image)[np.newaxis, ...]

        gt = dataframe[dataframe["id"] == image_path].iloc[0]['label']
        
        try:
            prediction = musiq.run_model_single_image(image_path)
        except Exception as e:
            print(f"Error predicting {image_path}: {e}")
            continue

        if prediction is None:
            continue

        gt_mean = get_weighted_mean(gt)
        
        count_correct += 1 if prediction > MUSIQ_MEAN and gt_mean > GT_MEAN or prediction < MUSIQ_MEAN and gt_mean < GT_MEAN else 0
        count_all += 1

        if count_all == TEST_SIZE:
            break

    print(f"Count correct: {count_correct}")
    print(f"Count all: {count_all}")
    print(f"Accuracy: {count_correct / count_all}")


if __name__ == "__main__":
    evaluate_musiq_high_low(AVA_DATASET_TEST_PATH)
    
