
import sys
import pandas as pd
import os
import numpy as np
import cv2
from tqdm import tqdm
from absl import flags
import matplotlib.pyplot as plt

from src.musiq.siamese_transformer_base import SiameseTransformerBase

IMAGE_FOLDER_PATH = "datasets/distortions/variants2/underexposure"

BATCH_SIZE = 1

MUSIQ_MEAN = 63.94
GT_MEAN = 5.40

TEST_SIZE = 100

"""
Evaluation on 100 images each distortion 

Weights: KoniQ-10k
Type -> Distortion Comparison, Mean, Median, Std
ORIGINAL -> _, 60.60, 63.95, 12.44
BLOB_OVERLAY -> 46% accuracy, 61.13, 63.94, 11.63
BLUR -> 100% accuracy, 21.33, 20.51, 5.23
ZOOM -> 79% accuracy, 55.64, 57.84, 12.43
ROTATION -> 83% accuracy, 56.47, 58.70, 11,27
SHEAR -> 90% accuracy, 52.74, 53.13, 12.87
OVEREXPOSURE -> 93% accuracy, 48.97, 49.97, 11.11
UNDEREXPOSURE -> 68% accuracy, 62.30, 64.20, 10.94
TOTAL -> 79.85% accuracy

Weights: PaQ-2-PiQ
ORIGINAL -> _, 70.53, 72.14, 5.74
BLOB_OVERLAY -> _, 70.82, 72.26, 5.24
BLUR -> _, 57.12, 57.13, 4.30
ZOOM -> _, 69.23, 70.81, 6.17
ROTATION -> _, 65.70, 69.68, 5.66
SHEAR -> _, 63.97, 68.08, 10.29
OVEREXPOSURE -> _, 68.47, 69.80, 5.53
UNDEREXPOSURE -> _, 70.62, 72.12, 5.70

Weights: SPAQ
ORIGINAL -> _, 63.24, 68.28, 13.83
BLOB_OVERLAY -> _, 63.37, 68.28, 15.50
BLUR -> _, 29.77, 29.69, 5.66
ZOOM -> _, 57.70, 58.69, 15.47
ROTATION -> _, 59.89, 63.96, 15.44
SHEAR -> _, 51.13, 51.72, 18.68
OVEREXPOSURE -> _, 50.38, 52.50, 16.62
UNDEREXPOSURE -> _, 64.55, 67.28, 13.22
"""

MODEL_CKPT = 'models/spaq_ckpt.npz'


def get_histograms():
    musiq = SiameseTransformerBase(ckpt_path=MODEL_CKPT)

    count_all = 0
    
    scores = []

    for image_filename in tqdm(os.listdir(IMAGE_FOLDER_PATH)):

        image_path = os.path.join(IMAGE_FOLDER_PATH, image_filename)

        try:
            score = musiq.run_model_single_image(image_path)
        except Exception as e:
            print(f"Error: {e}")
            continue

        scores.append(score)


        count_all += 1

        if count_all == TEST_SIZE:
            break

    print("Mean: {}".format(np.mean(scores)))
    print("Median: {}".format(np.median(scores)))
    print("Std: {}".format(np.std(scores)))
    
    # Creating the histogram
    plt.hist(scores, bins=10, color='#8B0000', edgecolor='black', alpha=0.7)

    # Adding whiskers
    quartiles = np.percentile(scores, [25, 50, 75])
    whisker_width = 1.5
    whisker_range = [quartiles[0] - whisker_width * (quartiles[2] - quartiles[0]),
                    quartiles[2] + whisker_width * (quartiles[2] - quartiles[0])]
    plt.axvline(whisker_range[0], color='red', linestyle='--')
    plt.axvline(whisker_range[1], color='red', linestyle='--')

    # Adding mean and median lines
    plt.axvline(np.mean(scores), color='green', linestyle='-', linewidth=2)
    plt.axvline(np.median(scores), color='blue', linestyle='-', linewidth=2)

    plt.xlabel('Mean-Opinion Score')
    plt.ylabel('Frequency')
    plt.title('Histogram of MOS; Variant: Underexposure')
    plt.xlim(0, 100)
    plt.show()



if __name__ == "__main__":
    get_histograms()
    
