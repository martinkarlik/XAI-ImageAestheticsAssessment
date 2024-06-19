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

IAA_MODEL = "models/nima_loss-0.078.hdf5"
# IMAGE_PATH = "datasets/evaluation-groups/plant/g_1_44.34.jpeg"
IMAGE_PATH = "datasets/evaluation-groups/plant/g_2_38.69.jpeg"


if __name__ == "__main__":


    image= cv2.resize(cv2.imread(IMAGE_PATH), (480, 640)) / 255.0
    image_rgb = cv2.cvtColor(np.uint8(image * 255.0), cv2.COLOR_BGR2RGB) / 255.0
    plt.figure("Original image")
    plt.imshow(image_rgb)
    plt.show()

    mask = np.zeros((640, 480), dtype=np.uint8)

    canvas = image_rgb.copy()

    def draw_mask(event, x, y, flags, param):
        global mask
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(mask, (x, y), 5, 255, -1) 

    cv2.namedWindow('Draw Mask')
    cv2.setMouseCallback('Draw Mask', draw_mask)


    

    while True:
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        canvas_with_mask = cv2.addWeighted(canvas, 0.8, mask_bgr / 255, 0.2, 0)
        cv2.imshow('Draw Mask', canvas_with_mask)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()

    max_value = np.amax(mask)
    print(max_value)

    heatmap = np.uint8(gray2rgb(mask))
    smoothed_heatmap = gaussian(heatmap, sigma=1)
    binary_heatmap = np.where(smoothed_heatmap > 0.5, 1, 0)
    binary_heatmap = np.squeeze(binary_heatmap[:, :, 0])
    green_color = [0, 255.0, 0]  # Green color
    red_color = [255, 0.0, 0]     # Red color
    overlay = np.zeros_like(image_rgb, dtype=np.uint8)
    overlay[binary_heatmap == 1.0] = green_color  # Overlay green where heatmap is white
    overlay[binary_heatmap == 0.0] = red_color      # Overlay red where heatmap is black

    blended_image = cv2.addWeighted(image_rgb, 0.8, overlay / 255, 0.2, 0)

    # Display the blended image
    plt.figure("Blended Image")
    plt.imshow(blended_image)
    plt.axis('off')
    plt.show()



