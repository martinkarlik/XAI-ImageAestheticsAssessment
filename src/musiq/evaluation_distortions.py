
import sys
import pandas as pd
import os
import numpy as np
import cv2
from tqdm import tqdm
from absl import flags

from src.musiq.siamese_transformer_base import SiameseTransformerBase

ORIGINALS_PATH = "datasets/distortions/variants2/original"
DISTORTIONS_PATH = "datasets/distortions/variants2/underexposure"

BATCH_SIZE = 1

MUSIQ_MEAN = 63.94
GT_MEAN = 5.40

TEST_SIZE = 100

def evaluate_musiq_distortions():
    musiq = SiameseTransformerBase()

    count_all = 0
    count_correct = 0

    distorted_images = {}
    for image_filename in os.listdir(DISTORTIONS_PATH):

        image_index = image_filename.split("_")[1]
        distorted_images[image_index] = os.path.join(DISTORTIONS_PATH, image_filename)

    for i, image_filename in enumerate(tqdm(os.listdir(ORIGINALS_PATH))):

        image_index = image_filename.split("_")[1]
        
        distorted_image_path = distorted_images[image_index]
        original_image_path = os.path.join(ORIGINALS_PATH, image_filename)

        try:
            score_original = musiq.run_model_single_image(original_image_path)
            score_distorted = musiq.run_model_single_image(distorted_image_path)
        except Exception as e:
            print(f"Error: {e}")
            continue

        print(f"Original image score: {score_original}")
        print(f"Distorted image score: {score_distorted}")

        if score_original > score_distorted:
            count_correct += 1

        # Predict the original and distorted images and compare the scores
        count_all += 1

        if count_all == TEST_SIZE:
            break

    print(f"Count correct: {count_correct}")
    print(f"Count all: {count_all}")
    print(f"Accuracy: {count_correct / count_all}")


if __name__ == "__main__":
    evaluate_musiq_distortions()
    
