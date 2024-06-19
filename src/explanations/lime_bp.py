import tensorflow as tf
import random
import numpy as np
import os
import cv2
from lime import lime_image
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.color import gray2rgb
from skimage.filters import gaussian
from src.brightness_predictor.base_module_bp import BaseModuleBP

MODEL_PATH = "models/bp_0.008.hdf5"
IMAGES_PATH = "datasets/brightness-set/specific-image"

def predict_for_lime(images):

    predictions = [bp.bp_model.predict(np.asarray(image)[np.newaxis, ...]) for image in images]

    predictions_formatted = []
    for prediction in predictions:
        predictions_formatted.append([prediction[0][0], 1.0 - prediction[0][0]])

    return predictions_formatted

if __name__ == "__main__":
    bp = BaseModuleBP(custom_weights=MODEL_PATH)
    bp.build()
    bp.compile()

    for filename in os.listdir(IMAGES_PATH):

        if not filename.endswith(".jpg"):
            continue

        file = os.path.join(IMAGES_PATH, filename)

        image_to_explain = cv2.resize(cv2.imread(file), (224, 224)) / 255.0

        image_to_infer = np.asarray(image_to_explain)[np.newaxis, ...]
        outputs = bp.bp_model.predict(image_to_infer)
        print("{}: {}".format(file, outputs))

        explainer = lime_image.LimeImageExplainer()

        image_rgb = cv2.cvtColor(np.uint8(image_to_explain * 255.0), cv2.COLOR_BGR2RGB)

        plt.figure("Original image")
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.show()

        # Explain the prediction on this image
        explanation = explainer.explain_instance(image_to_explain,
                                                    lambda x: predict_for_lime(x),
                                                    hide_color=0,
                                                    num_samples=500)


        _, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, negative_only=False, num_features=2, hide_rest=True)

        heatmap = np.uint8(gray2rgb(mask))
        smoothed_heatmap = gaussian(heatmap, sigma=1)
        binary_heatmap = np.where(smoothed_heatmap > 0.5, 1, 0)
        binary_heatmap = np.squeeze(binary_heatmap[:, :, 0])

        green_color = [0, 255.0, 0]  # Green color
        red_color = [255, 0.0, 0]     # Red color
        overlay = np.zeros_like(image_rgb, dtype=np.uint8)
        overlay[binary_heatmap == 1.0] = green_color  # Overlay green where heatmap is white
        overlay[binary_heatmap == 0.0] = red_color      # Overlay red where heatmap is black


        # Blend the overlay with the original image
        blended_image = cv2.addWeighted(image_rgb / 255.0, 0.8, overlay / 255.0, 0.2, 0)

        # Display the blended image
        plt.figure("Blended Image")
        plt.imshow(blended_image)
        plt.axis('off')
        plt.show()

        # Black and White explanation
        # heatmap = np.uint8(gray2rgb(mask))
        # smoothed_heatmap = gaussian(heatmap, sigma=1)

        # plt.figure("Heatmap")
        # plt.imshow(smoothed_heatmap, cmap='hot')
        # plt.axis('off')
        # plt.show()