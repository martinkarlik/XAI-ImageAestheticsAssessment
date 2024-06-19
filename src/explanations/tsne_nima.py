import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import cv2
from mpl_toolkits.mplot3d import Axes3D

from src.nima.giiaa.base_module_giiaa import BaseModuleGIIAA
from src.nima.gciaa.base_module_gciaa import BaseModuleGCIAA

DATASET_PATH_ORIG = "datasets/distortions/variants_v2/original"
DISTORTED_IMAGES_PARENT_DIR = 'datasets/distortions/variants_v2'
# DISTORTION_NAMES = ['underexposure', 'overexposure', 'rotation', 'shear', 'zoom', 'blur', 'blob_overlay']
DISTORTION_NAMES = ['blob_overlay']
COLORS = ['b', 'g', 'c', 'm', 'y', 'k', '#A020F0']

# WEIGHTS = "models/nima_loss-0.078.hdf5"
WEIGHTS = "models/blob_overlay_0.927.hdf5"

if __name__ == "__main__":

    nima = BaseModuleGCIAA(custom_weights=WEIGHTS)
    nima.build()

    model_to_use = nima.siamese_model_unpacked
    # model_to_use = nima.nima_model_unpacked

    print(model_to_use.summary())

    features = np.empty((640, 1536))
    for i, image_filename in tqdm(enumerate(os.listdir(DATASET_PATH_ORIG))):
        image_path = os.path.join(DATASET_PATH_ORIG, image_filename)
        image = cv2.resize(cv2.imread(image_path), (224, 224)) / 255.0
        image = np.asarray(image)[np.newaxis, ...]
        prediction = model_to_use.predict(image)[0]
        features[i, :] = prediction

    # Apply t-SNE to reduce dimensionality
    tsne = TSNE(n_components=2, perplexity=80, random_state=42)
    features_tsne = tsne.fit_transform(features)

    plt.scatter(features_tsne[:, 0], features_tsne[:, 1], marker='.', color='r', label='original')

    for j, distortion_name in enumerate(DISTORTION_NAMES):

        if j > 1:
            break

        features = np.empty((640, 1536))
        distorted_images_dir = os.path.join(DISTORTED_IMAGES_PARENT_DIR, distortion_name)
        for i, image_filename in enumerate(os.listdir(distorted_images_dir)):
            image_path = os.path.join(distorted_images_dir, image_filename)
            image = cv2.resize(cv2.imread(image_path), (224, 224)) / 255.0
            image = np.asarray(image)[np.newaxis, ...]
            prediction = model_to_use.predict(image)[0]
            features[i, :] = prediction


        # Apply t-SNE to reduce dimensionality
        tsne = TSNE(n_components=2, perplexity=80, random_state=42)
        features_tsne = tsne.fit_transform(features)

        plt.scatter(features_tsne[:, 0], features_tsne[:, 1], marker='.', color=COLORS[j], label=distortion_name)


    # Visualize the t-SNE embedding
    plt.title('t-SNE Visualization of NIMA Features')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()
