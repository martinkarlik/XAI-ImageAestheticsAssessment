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

GCIAA_SMILES_MODEL = "models/smiles_0.982.hdf5"
SMILES_FOLDER = "datasets/portraits/smile_test"
FROWNS_FOLDER = "datasets/portraits/frown_test"


if __name__ == "__main__":

    # Compile GCIAA model
    gciaa = BaseModuleGCIAA(custom_weights=GCIAA_SMILES_MODEL, load_weights_as='GCIAA')
    gciaa.build()
    gciaa.compile()


    gciaa_nima = BaseModuleGCIAA(custom_weights=GCIAA_NIMA_MODEL, load_weights_as='GIIAA')
    gciaa_nima.build()
    gciaa_nima.compile()

    for i in range(10):

        frowns_image_filename = random.choice(os.listdir(FROWNS_FOLDER))
        frowns_image_path = os.path.join(FROWNS_FOLDER, frowns_image_filename)
        frowns_image = cv2.resize(cv2.imread(frowns_image_path), (224, 224)) / 255.0
        frowns_image_np = np.asarray(frowns_image)[np.newaxis, ...]

        smiles_image_filename = random.choice(os.listdir(SMILES_FOLDER))
        smiles_image_path = os.path.join(SMILES_FOLDER, smiles_image_filename)
        smiles_image = cv2.resize(cv2.imread(smiles_image_path), (224, 224)) / 255
        smiles_image_np = np.asarray(smiles_image)[np.newaxis, ...]

        # Make inference with GCIAA Smiles model
        gciaa_prediction = gciaa.predict([frowns_image_np, smiles_image_np])[0, 0]
        print("Comparison Smiles: {:.2f}".format(gciaa_prediction))

        frowns_image_prediction = gciaa.predict_image_encoder(frowns_image_np)[0]
        smiles_image_prediction = gciaa.predict_image_encoder(smiles_image_np)[0]

        indices = np.arange(len(frowns_image_prediction))
        width = 0.35
        plt.figure(figsize=(10, 6))
        plt.bar(indices, frowns_image_prediction, width, label='Frown Image Prediction', alpha=0.7)
        plt.bar(indices + width, smiles_image_prediction, width, label='Smile Image Prediction', alpha=0.7)

        cv2.imshow('Image of a Frown', frowns_image)
        cv2.imshow('Image of a Smile', smiles_image)

        plt.title('Distribution of Predicted Scores using Smiles-trained model')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')
        plt.show()


        # Make inference with GCIAA Base model
        gciaa_prediction = gciaa.predict([frowns_image_np, smiles_image_np])[0, 0]
        print("Comparison Base: {:.2f}".format(gciaa_prediction))

        frowns_image_prediction = gciaa_nima.predict_image_encoder(frowns_image_np)[0]
        smiles_image_prediction = gciaa_nima.predict_image_encoder(smiles_image_np)[0]

        indices = np.arange(len(frowns_image_prediction))
        width = 0.35
        plt.figure(figsize=(10, 6))
        plt.bar(indices, frowns_image_prediction, width, label='Frown Image Prediction', alpha=0.7)
        plt.bar(indices + width, smiles_image_prediction, width, label='Smile Image Prediction', alpha=0.7)

        cv2.imshow('Image of a Frown', frowns_image)
        cv2.imshow('Image of a Smile', smiles_image)

        plt.title('Distribution of Predicted Scores using base model')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')
        plt.show()

