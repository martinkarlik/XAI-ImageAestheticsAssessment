import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import cv2
import collections
import io
import os
from typing import Dict, Sequence, Text, TypeVar
from absl import app
from absl import flags
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import tensorflow.compat.v1 as tf

import src.musiq.multiscale_transformer as model_mod
import src.musiq.preprocessing as pp_lib
from src.musiq.siamese_transformer_base import SiameseTransformerBase

DATASET_PATH_ORIG = "datasets/distortions/originals/original"
DISTORTED_IMAGES_PARENT_DIR = 'datasets/distortions/variants'
DISTORTION_NAMES = ['underexposure', 'overexposure', 'rotation', 'shear', 'zoom', 'blur', 'blob_overlay']
COLORS = ['b', 'g', '#7c3f00', 'c', 'm', 'y', 'k']

WEIGHTS = 'models/koniq_ckpt.npz'

DETASET_SIZE = 640
# FEATURE_SIZE = 1536
FEATURE_SIZE = 384

if __name__ == "__main__":

    musiq = SiameseTransformerBase(ckpt_path=WEIGHTS)

    count_errors = 0

    features = None
    for i, image_filename in tqdm(enumerate(os.listdir(DATASET_PATH_ORIG))):
        try:
            image_path = os.path.join(DATASET_PATH_ORIG, image_filename)
            prediction = musiq.run_model_unpacked(image_path)
            if features is None:
                features = prediction
            else:
                features = np.vstack((features, prediction))
        except Exception as e:
            count_errors += 1
            print(f"Error: {e}")

    print("Original; errors: ", count_errors)

    # Apply t-SNE to reduce dimensionality
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_tsne = tsne.fit_transform(features)

    plt.scatter(features_tsne[:, 0], features_tsne[:, 1], marker='.', color='r', label='original')

    for j, distortion_name in enumerate(DISTORTION_NAMES):

        count_errors = 0
        features = None
        distorted_images_dir = os.path.join(DISTORTED_IMAGES_PARENT_DIR, distortion_name)
        for i, image_filename in enumerate(os.listdir(distorted_images_dir)):
            try:
                image_path = os.path.join(distorted_images_dir, image_filename)
                prediction = musiq.run_model_unpacked(image_path)
                if features is None:
                    features = prediction
                else:
                    features = np.vstack((features, prediction))
            except Exception as e:
                count_errors += 1
                print(f"Error: {e}")

        print(f"{distortion_name}; errors: ", count_errors)

        # Apply t-SNE to reduce dimensionality
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        features_tsne = tsne.fit_transform(features)

        plt.scatter(features_tsne[:, 0], features_tsne[:, 1], marker='.', color=COLORS[j], label=distortion_name)


    # Visualize the t-SNE embedding
    plt.title('t-SNE Visualization of MusiQ Features')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()