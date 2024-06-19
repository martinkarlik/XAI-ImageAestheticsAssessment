import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage.color import gray2rgb
from skimage.filters import gaussian
from PIL import Image
import numpy as np
import os
import cv2

from src.nima.gciaa.base_module_gciaa import BaseModuleGCIAA
from src.nima.giiaa.base_module_giiaa import BaseModuleGIIAA
import matplotlib.pyplot as plt

# MODEL = "models/black_and_white_v2_0.939.hdf5"
# MODEL = "models/blob_overlay_0.956.hdf5"
# MODEL = "models/blur_v3_0.998.hdf5"
MODEL = "models/smiles_0.982.hdf5"
DATASET_TO_EXPLAIN = "datasets/visualize-perturbations"

def get_mean_as_probability(distribution):

    mean = 0.0
    for i in range(0, len(distribution)):
        mean += distribution[i] * (i + 1)
    return mean / 10.0


def predict_for_lime(images):

    for image in images:
        cv2.imshow("Image", image)
        cv2.waitKey(0)


    predictions = gciaa.image_encoder_model.predict(images)

    predictions_normalized = []
    for prediction in predictions:
        prob = get_mean_as_probability(prediction) / 10.0
        predictions_normalized.append([prob, 1.0 - prob])

    return predictions_normalized


if __name__ == "__main__":

    # Compile GIIAA model
    gciaa = BaseModuleGCIAA(custom_weights=MODEL, load_weights_as='GCIAA')
    gciaa.build()
    gciaa.compile()

    # Load all the images from a directory to explain
    images_to_explain = []
    for image_path in os.listdir(DATASET_TO_EXPLAIN):
        if not (image_path.endswith(".jpeg") or image_path.endswith(".jpg") or image_path.endswith(".png")):
            continue

        full_image_path = os.path.join(DATASET_TO_EXPLAIN, image_path)
        image = cv2.resize(cv2.imread(full_image_path), (224, 224)) / 255.0
        images_to_explain.append(image)

    explainer = lime_image.LimeImageExplainer()
    for image_to_explain in images_to_explain:

        image_rgb = cv2.cvtColor(np.uint8(image_to_explain * 255.0), cv2.COLOR_BGR2RGB) / 255.0
        plt.figure("Original image")
        plt.imshow(image_rgb)
        plt.show()

        # Explain the prediction on this image
        explanation_giiaa = explainer.explain_instance(image_to_explain,
                                                    lambda x: predict_for_lime(x),
                                                    num_samples=500)
        
        cv2.destroyAllWindows()


