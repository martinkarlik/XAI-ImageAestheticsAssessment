
import sys
import pandas as pd
import os
import numpy as np
import cv2
from tqdm import tqdm
from absl import flags
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from src.nima.giiaa.base_module_giiaa import BaseModuleGIIAA

GIIAA_PATH = "models/nima_loss-0.078.hdf5"

ORIGINAL_IMAGES_PATH = "datasets/distortions/variants_v2/original"
DISTORTED_IMAGES_PATH = "datasets/distortions/variants_v2/underexposure"

BATCH_SIZE = 1

MUSIQ_MEAN = 63.94
GT_MEAN = 5.40

TEST_SIZE = 640

"""
Evaluation on 100 images, each distortion, getting the mean of the NIMA distribution

Type -> Distortion Comparison, Mean, Median, Std
ORIGINAL -> _, 4.83, 4.84, 0.38 
BLOB_OVERLAY -> 63.0% (403/640), 4.83, 4.80, 0.36
BLUR -> 64.8% (415/640), 4.37, 4.34, 0.26
ZOOM -> 67.5% (432/640), 4.69, 4.76, 0.41
ROTATION -> 72.5% (464/640), 4.77, 4.79, 0.40
SHEAR -> 51.7% (331/640), 4.95, 5.00, 0.41
OVEREXPOSURE -> 64.5% (413/640), 4.80, 4.80, 0.33
UNDEREXPOSURE -> 39.5% (253/640), 4.90, 4.93, 0.35

___

Evaluation on 100 images, each distortion, getting the mode of the NIMA distribution

Type -> Distortion Comparison, Mean, Median, Std
ORIGINAL -> _, 4.88, 5.0, 0.32
BLOB_OVERLAY -> _, 4.91, 5.0, 0.32
BLUR -> _, 4.44, 4.0, 0.50
ZOOM -> _, 4.78, 5.0, 0.41
ROTATION -> _, 4.83, 5.0, 0.38
SHEAR -> _, 4.89, 5.0, 0.34
OVEREXPOSURE -> _, 4.92, 5.0, 0.27
UNDEREXPOSURE -> _, 4.92, 5.0, 0.27

"""

def get_weighted_mean(distribution):
    mean = 0.0
    for i in range(0, len(distribution)):
        mean += distribution[i] * (i + 1)
    return mean

def get_mode(distribution):
    mode_index = np.argmax(distribution)
    mode = mode_index + 1  # Since ratings start from 1
    return mode

def find_matching_image_path(filename, folder):
    index = filename.split('_')[1]

    for file in os.listdir(folder):
        if file.startswith('_' + index):
            return os.path.join(folder, file)

def get_histograms():
    nima = BaseModuleGIIAA(custom_weights=GIIAA_PATH)
    nima.build()
    nima.compile()

    count_all = 0
    count_correct = 0
    
    # scores = []

    for image_filename in tqdm(os.listdir(ORIGINAL_IMAGES_PATH)):

        orig_image_path = os.path.join(ORIGINAL_IMAGES_PATH, image_filename)
        orig_image = cv2.resize(cv2.imread(orig_image_path), (224, 224)) / 255.0
        orig_image = np.asarray(orig_image)[np.newaxis, ...]

        distorted_image_path = find_matching_image_path(image_filename, DISTORTED_IMAGES_PATH)
        distorted_image = cv2.resize(cv2.imread(distorted_image_path), (224, 224)) / 255.0
        distorted_image = np.asarray(distorted_image)[np.newaxis, ...]

        try:
            prediction = nima.nima_model.predict(orig_image)[0]
            score_a = get_weighted_mean(prediction)

            prediction = nima.nima_model.predict(distorted_image)[0]
            score_b = get_weighted_mean(prediction)
        except Exception as e:
            print(f"Error: {e}")
            continue


        count_all += 1
        count_correct += 1 if score_a >= score_b else 0

        if count_all == TEST_SIZE:
            break

    print("Count correct: {}".format(count_correct))
    print("Count all: {}".format(count_all))
    print("Accuracy: {}".format(count_correct / count_all))

    
    # print("Mean: {}".format(np.mean(scores)))
    # print("Median: {}".format(np.median(scores)))
    # print("Std: {}".format(np.std(scores)))

    # # Creating the histogram
    # plt.hist(scores, bins=10, color='#006400', edgecolor='black', alpha=0.7)

    # # Adding whiskers
    # quartiles = np.percentile(scores, [25, 50, 75])
    # whisker_width = 1.5
    # whisker_range = [quartiles[0] - whisker_width * (quartiles[2] - quartiles[0]),
    #                 quartiles[2] + whisker_width * (quartiles[2] - quartiles[0])]
    # plt.axvline(whisker_range[0], color='red', linestyle='--')
    # plt.axvline(whisker_range[1], color='red', linestyle='--')

    # # Adding mean and median lines
    # plt.axvline(np.mean(scores), color='green', linestyle='-', linewidth=2)
    # plt.axvline(np.median(scores), color='blue', linestyle='-', linewidth=2)

    # plt.xlabel('Mode-Opinion Score')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Mode-Opinion Scores; Variant: Underexposure')
    # plt.xlim(0, 10)
    # plt.show()



if __name__ == "__main__":
    get_histograms()
    
    # scores = [2, 7, 27, 40, 65, 48, 32, 20, 8, 4]
    # plt.bar(range(1, 11), scores, color='green', edgecolor='black')
    # plt.xlabel('Rating')
    # plt.ylabel('Votes')
    # plt.title('Distribution of user scores for the given image')
    # plt.show()


