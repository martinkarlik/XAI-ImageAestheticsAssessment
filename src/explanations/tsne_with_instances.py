import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import cv2
import math
from mpl_toolkits.mplot3d import Axes3D

from src.nima.giiaa.base_module_giiaa import BaseModuleGIIAA
from src.nima.gciaa.base_module_gciaa import BaseModuleGCIAA

NUM_OF_IMAGES = 640
DATASET_PATH_ORIG = "datasets/distortions/variants_v2/original"
DATASET_PATH_DISTORTED = "datasets/distortions/variants_v2/bw"
HANDPICKED_IMAGES = ["139", "579", "439"]

# NUM_OF_IMAGES = 30
# DATASET_PATH_ORIG = "datasets/distortions/variants_v3/original"
# DATASET_PATH_DISTORTED = "datasets/distortions/variants_v3/blur"
# HANDPICKED_IMAGES = ["6", "610", "630"]


WEIGHTS_BASE = "models/nima_loss-0.078.hdf5"
WEIGHTS_FINE_TUNED = "models/blob_overlay_0.956.hdf5"
# WEIGHTS_FINE_TUNED = "models/black_and_white_v2_0.939.hdf5"
# WEIGHTS_FINE_TUNED = "models/blur_0.995.hdf5"

# COLOURS_ORIG = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
# COLOURS_DISTORTED = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']

def visualize_complete_tsne(weights, load_weights_as, model_name, labels, image_names_stored=None):
    nima = BaseModuleGCIAA(custom_weights=weights, load_weights_as=load_weights_as)
    nima.build()
    # model_to_use = nima.siamese_model_unpacked
    model_to_use = nima.image_encoder_model_unpacked

    colors_a = ['#A62309', '#E12311', '#EE8B78']
    colors_b = ['#0A0882', '#2824D5', '#817FE4']


    if image_names_stored is None:
        split_image_names = {}
        split_image_names["original"] = construct_tsne(model_to_use, DATASET_PATH_ORIG, colors_a, labels[0], num_images=NUM_OF_IMAGES)
        split_image_names["distorted"] = construct_tsne(model_to_use, DATASET_PATH_DISTORTED, colors_b, labels[1], num_images=NUM_OF_IMAGES)
    else:
        _ = construct_tsne(model_to_use, DATASET_PATH_ORIG, colors_a, labels[0], num_images=NUM_OF_IMAGES, image_names_stored=image_names_stored["original"])
        _ = construct_tsne(model_to_use, DATASET_PATH_DISTORTED, colors_b, labels[1], num_images=NUM_OF_IMAGES, image_names_stored=image_names_stored["distorted"])

    # Visualize the t-SNE embedding
    plt.title('t-SNE Visualization of {} model'.format(model_name))
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()

    return split_image_names

def construct_tsne(model_to_use, dataset_path, colors, label, num_images=640, num_features=1536, image_names_stored=None):
    features = np.empty((num_images, num_features))
    image_names = np.empty(num_images)

    valid_image_names = [image for image in os.listdir(dataset_path) if not image.startswith('.')]
    for i, image_filename in tqdm(enumerate(valid_image_names)):
        image_path = os.path.join(dataset_path, image_filename)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224)) / 255.0
        image = np.asarray(image)[np.newaxis, ...]
        prediction = model_to_use.predict(image)[0]
        features[i, :] = prediction
        image_names[i] = image_filename.split('_')[1]


    # Apply t-SNE to reduce dimensionality
    tsne = TSNE(n_components=2, perplexity=80, random_state=42)
    features_tsne = tsne.fit_transform(features)

    features_tsne_a = []
    features_tsne_b = []
    features_tsne_c = []

    max_x = max(abs(features_tsne[:, 0]))
    max_y = max(abs(features_tsne[:, 1]))
    # max_point_euclidean = math.sqrt(max_x ** 2 + max_y ** 2)

    split_image_names = []

    if not image_names_stored:

        image_names_a = []
        image_names_b = []
        image_names_c = []

        for i, feature in enumerate(features_tsne):
            # this_feature_euclidean = math.sqrt(feature[0] ** 2 + feature[1] ** 2)

            # if this_feature_euclidean < max_point_euclidean / 3:
            #     features_tsne_a.append([feature[0], feature[1]])
            #     image_names_a.append(image_names[i])
            # elif this_feature_euclidean < 2 * max_point_euclidean / 3:
            #     features_tsne_b.append([feature[0], feature[1]])
            #     image_names_b.append(image_names[i])
            # else:
            #     features_tsne_c.append([feature[0], feature[1]])
            #     image_names_c.append(image_names[i])

            if abs(feature[0]) < max_x / 3 and abs(feature[1]) < max_y / 3:
                features_tsne_a.append([feature[0], feature[1]])
                image_names_a.append(image_names[i])
            elif abs(feature[0]) < 2 * max_x / 3 and abs(feature[1]) < 2 * max_y / 3:
                features_tsne_b.append([feature[0], feature[1]])
                image_names_b.append(image_names[i])
            else:
                features_tsne_c.append([feature[0], feature[1]])
                image_names_c.append(image_names[i])

        split_image_names = [image_names_a, image_names_b, image_names_c]

    else:
        image_names_a = image_names_stored[0]
        image_names_b = image_names_stored[1]
        image_names_c = image_names_stored[2]

        for i, feature in enumerate(features_tsne):
            if image_names[i] in image_names_a:
                features_tsne_a.append([feature[0], feature[1]])
            elif image_names[i] in image_names_b:
                features_tsne_b.append([feature[0], feature[1]])
            else:
                features_tsne_c.append([feature[0], feature[1]])

    features_tsne_a = np.array(features_tsne_a)
    features_tsne_b = np.array(features_tsne_b)
    features_tsne_c = np.array(features_tsne_c)
        
    plt.scatter(features_tsne_a[:, 0], features_tsne_a[:, 1], marker='.', color=colors[0], label=label)
    plt.scatter(features_tsne_b[:, 0], features_tsne_b[:, 1], marker='.', color=colors[1])
    plt.scatter(features_tsne_c[:, 0], features_tsne_c[:, 1], marker='.', color=colors[2])

    for i in range(0, NUM_OF_IMAGES):
        name = str(int(image_names[i]))
        if name in HANDPICKED_IMAGES:
            plt.annotate(name, (features_tsne[i, 0] + 0.1, features_tsne[i, 1] + 0.1), fontsize=4, ha='right', va='bottom')

    return split_image_names

if __name__ == "__main__":

    split_image_names = visualize_complete_tsne(WEIGHTS_BASE, 'GIIAA', model_name='Blob-Overlay-trained', labels=['original', 'blob-overlay'])
    visualize_complete_tsne(WEIGHTS_FINE_TUNED, 'GCIAA', model_name='base', labels=['original', 'blob-overlay'], image_names_stored=split_image_names)
