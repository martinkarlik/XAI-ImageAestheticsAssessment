import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import cv2
import re
from mpl_toolkits.mplot3d import Axes3D

from src.nima.giiaa.base_module_giiaa import BaseModuleGIIAA
from src.nima.gciaa.base_module_gciaa import BaseModuleGCIAA

# NUM_OF_IMAGES = 640
# DATASET_PATH_ORIG = "datasets/distortions/variants_v2/original"
# DATASET_PATH_DISTORTED = "datasets/distortions/variants_v2/blob_overlay_v2"
# HANDPICKED_IMAGES = ["139", "579", "439"]

NUM_OF_IMAGES = 20
DATASET_PATH_SMILES = "datasets/portraits/smile_test"
DATASET_PATH_FROWNS = "datasets/portraits/frown_test"
HANDPICKED_IMAGES = ["3", "5", "11", "16"]


# WEIGHTS = "models/nima_loss-0.078.hdf5"
WEIGHTS = "models/smiles_0.982.hdf5"

if __name__ == "__main__":

    nima = BaseModuleGCIAA(custom_weights=WEIGHTS, load_weights_as='GCIAA')
    nima.build()

    # model_to_use = nima.siamese_model_unpacked
    model_to_use = nima.image_encoder_model_unpacked

    smile_image_names = [image for image in os.listdir(DATASET_PATH_SMILES) if not image.startswith('.')]
    frown_image_names = [image for image in os.listdir(DATASET_PATH_FROWNS) if not image.startswith('.')]


    features = np.empty((NUM_OF_IMAGES, 1536))
    image_names = np.empty(NUM_OF_IMAGES)
    for i, image_filename in tqdm(enumerate(smile_image_names)):
        image_path = os.path.join(DATASET_PATH_SMILES, image_filename)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224)) / 255.0
        image = np.asarray(image)[np.newaxis, ...]
        prediction = model_to_use.predict(image)[0]
        features[i, :] = prediction
        image_names[i] = re.search(r'\d+', image_filename).group()


    # Apply t-SNE to reduce dimensionality
    tsne = TSNE(n_components=2, perplexity=15, random_state=42)
    features_tsne = tsne.fit_transform(features)

    plt.scatter(features_tsne[:, 0], features_tsne[:, 1], marker='.', color='r', label='smiles')

    for i in range(0, NUM_OF_IMAGES):
        name = str(int(image_names[i]))
        if name in HANDPICKED_IMAGES:
            plt.annotate(name, (features_tsne[i, 0] + 0.1, features_tsne[i, 1] + 0.1), fontsize=4, ha='right', va='bottom')

    features = np.empty((NUM_OF_IMAGES, 1536))
    image_names = np.empty(NUM_OF_IMAGES)
    for i, image_filename in enumerate(frown_image_names):
        image_path = os.path.join(DATASET_PATH_FROWNS, image_filename)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224)) / 255.0
        image = np.asarray(image)[np.newaxis, ...]
        prediction = model_to_use.predict(image)[0]
        features[i, :] = prediction
        image_names[i] = re.search(r'\d+', image_filename).group()


    # Apply t-SNE to reduce dimensionality
    tsne = TSNE(n_components=2, perplexity=15, random_state=42)
    features_tsne = tsne.fit_transform(features)

    plt.scatter(features_tsne[:, 0], features_tsne[:, 1], marker='.', color='b', label='frowns')

    for i in range(0, NUM_OF_IMAGES):
        name = str(int(image_names[i]))
        if name in HANDPICKED_IMAGES:
            plt.annotate(name, (features_tsne[i, 0] + 0.1, features_tsne[i, 1] + 0.1), fontsize=4, ha='right', va='bottom')


    # Visualize the t-SNE embedding
    plt.title('t-SNE Visualization of Smiles-trained model')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()
