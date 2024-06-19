import tensorflow_hub as hub
from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage.color import gray2rgb
from skimage.filters import gaussian
from PIL import Image
import numpy as np
import os
import cv2
import io
from openai import OpenAI

from src.nima.gciaa.base_module_gciaa import BaseModuleGCIAA
from src.nima.giiaa.base_module_giiaa import BaseModuleGIIAA
import matplotlib.pyplot as plt

# MODEL = "models/black_and_white_v2_0.939.hdf5"
# MODEL = "models/blob_overlay_0.956.hdf5"
# MODEL = "models/blur_v3_0.998.hdf5"
MODEL = "models/smiles_0.982.hdf5"
DATASET_TO_EXPLAIN = "datasets/visualize-perturbations"

STORED_PERTURBATIONS = "datasets/visualize-perturbations/stored-genai-perturbations"

client = OpenAI(api_key="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")

def numpy_to_bytes(image_array):
    """
    Convert a numpy array image to bytes suitable for OpenAI API.
    """
    pil_image = Image.fromarray((image_array * 255).astype(np.uint8)).convert('RGBA')
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr, format='PNG')
    byte_arr = byte_arr.getvalue()
    return byte_arr


def load_perturbations():
    perturbations = []

    for file in os.listdir(STORED_PERTURBATIONS):
        if not file.endswith(".png"):
            continue

        full_image_path = os.path.join(STORED_PERTURBATIONS, file)
        image = cv2.resize(cv2.imread(full_image_path), (224, 224)) / 255.0
        perturbations.append(image)

    return np.array(perturbations)

def get_mean_as_probability(distribution):

    mean = 0.0
    for i in range(0, len(distribution)):
        mean += distribution[i] * (i + 1)
    return mean / 10.0


def predict_for_lime(images):

    original_instance = images[0]
    perturbations = load_perturbations()

    predictions = []

    original_instance = np.asarray(original_instance)[np.newaxis, ...]
    original_score = gciaa.image_encoder_model.predict(original_instance)[0]
    predictions.append(original_score)

    print(perturbations.shape)
    for perturbation in perturbations:
        cv2.imshow("Perturbation", perturbation)
        cv2.waitKey()
        perturbation_instance = np.asarray(perturbation)[np.newaxis, ...]
        prediction = gciaa.image_encoder_model.predict(perturbation_instance)[0]
        predictions.append(prediction)

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
        explanation_giiaa = explainer.explain_instance(image_rgb,
                                                    lambda x: predict_for_lime(x),
                                                    num_samples=7)

        _, mask = explanation_giiaa.get_image_and_mask(explanation_giiaa.top_labels[0], positive_only=False, negative_only=False, num_features=1, hide_rest=True)

        heatmap = np.uint8(gray2rgb(mask * 255.0))
        smoothed_heatmap = gaussian(heatmap, sigma=1)
        binary_heatmap = np.where(smoothed_heatmap > 0.5, 1, 0)
        binary_heatmap = np.squeeze(binary_heatmap[:, :, 0])

        green_color = [0.0, 255.0, 0.0]  # Green color
        red_color = [255.0, 0.0, 0.0]     # Red color
        overlay = np.zeros_like(image_rgb, dtype=np.uint8)
        overlay[binary_heatmap == 1.0] = green_color  # Overlay green where heatmap is white
        overlay[binary_heatmap == 0.0] = red_color      # Overlay red where heatmap is black

        # Blend the overlay with the original image
        blended_image = cv2.addWeighted(image_rgb, 0.8, overlay / 255, 0.2, 0)

        # Display the blended image
        plt.figure("Blended Image")
        plt.imshow(blended_image)
        plt.axis('off')
        plt.show()



