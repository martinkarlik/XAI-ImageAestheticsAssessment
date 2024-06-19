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

IMAGE_PATH = "datasets/evaluation-groups/plant/g_2_38.69.jpeg"

def visualize_heatmap_difference(image_to_explain, heatmap_a, heatmap_b):
    smoothed_heatmap_a = gaussian(heatmap_a, sigma=1)
    binary_heatmap_a = np.where(smoothed_heatmap_a > 0.5, 1, 0)
    binary_heatmap_a = np.squeeze(binary_heatmap_a[:, :, 0])

    smoothed_heatmap_b = gaussian(heatmap_b, sigma=1)
    binary_heatmap_b = np.where(smoothed_heatmap_b > 0.5, 1, 0)
    binary_heatmap_b = np.squeeze(binary_heatmap_b[:, :, 0])

    heatmap_difference = binary_heatmap_b - binary_heatmap_a

    green_color = [0.0, 255.0, 0.0]  # Green color
    red_color = [255.0, 0.0, 0.0]     # Red color

    overlay = np.zeros((640, 480, 3), dtype=np.uint8)
    overlay[heatmap_difference == 1.0] = green_color 
    overlay[heatmap_difference == -1.0] = red_color
    overlay[heatmap_difference == 0.0] = image_to_explain[heatmap_difference == 0.0]

    image_rgb = cv2.cvtColor(np.uint8(image_to_explain * 255.0), cv2.COLOR_BGR2RGB) / 255.0
    blended_image = cv2.addWeighted(image_rgb, 0.8, overlay / 255, 0.2, 0)

    # Display the blended image
    plt.figure("Difference Explanation")
    plt.imshow(blended_image)
    plt.axis('off')
    plt.show()

def visualize_heatmap(image, heatmap):
    green_color = [0.0, 255.0, 0.0]  # Green color
    red_color = [255.0, 0.0, 0.0]     # Red color

    smoothed_heatmap = gaussian(heatmap, sigma=1)
    binary_heatmap = np.where(smoothed_heatmap > 0.5, 1, 0)
    binary_heatmap = np.squeeze(binary_heatmap[:, :, 0])

    green_color = [0.0, 255.0, 0.0]  # Green color
    red_color = [255.0, 0.0, 0.0]     # Red color
    overlay = np.zeros_like(image, dtype=np.uint8)
    overlay[binary_heatmap == 1.0] = green_color  # Overlay green where heatmap is white
    overlay[binary_heatmap == 0.0] = red_color      # Overlay red where heatmap is black

    image_rgb = cv2.cvtColor(np.uint8(image * 255.0), cv2.COLOR_BGR2RGB) / 255.0
    blended_image = cv2.addWeighted(image_rgb, 0.8, overlay / 255, 0.2, 0)

    # Display the blended image
    plt.figure("Explanation")
    plt.imshow(blended_image)
    plt.axis('off')
    plt.show()
    

if __name__ == "__main__":

    image = cv2.resize(cv2.imread(IMAGE_PATH), (480, 640)) / 255.0
    image_rgb = cv2.cvtColor(np.uint8(image * 255.0), cv2.COLOR_BGR2RGB) / 255.0
    plt.figure("Original image")
    plt.imshow(image_rgb)
    plt.show()

    # Mark-up general explanation
    mask_general = np.zeros((640, 480), dtype=np.uint8)
    canvas = image.copy()

    def draw_mask(event, x, y, flags, param):
        global mask_general
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(mask_general, (x, y), 10, 255, -1) 

    cv2.namedWindow('Draw Mask')
    cv2.setMouseCallback('Draw Mask', draw_mask)


    while True:
        mask_bgr = cv2.cvtColor(mask_general, cv2.COLOR_GRAY2BGR)
        canvas_with_mask = cv2.addWeighted(canvas, 0.8, mask_bgr / 255, 0.2, 0)
        cv2.imshow('Draw Mask', canvas_with_mask)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    
    # Mark-up personalized explanation
    mask_personalized = np.zeros((640, 480), dtype=np.uint8)
    canvas = image.copy()

    def draw_mask(event, x, y, flags, param):
        global mask_personalized
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(mask_personalized, (x, y), 10, 255, -1) 

    cv2.namedWindow('Draw Mask')
    cv2.setMouseCallback('Draw Mask', draw_mask)


    while True:
        mask_bgr = cv2.cvtColor(mask_personalized, cv2.COLOR_GRAY2BGR)
        canvas_with_mask = cv2.addWeighted(canvas, 0.8, mask_bgr / 255, 0.2, 0)
        cv2.imshow('Draw Mask', canvas_with_mask)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    heatmap_a = np.uint8(gray2rgb(mask_general))
    heatmap_b = np.uint8(gray2rgb(mask_personalized))
    visualize_heatmap(image=image, heatmap=heatmap_a)
    visualize_heatmap(image=image, heatmap=heatmap_b)
    visualize_heatmap_difference(image_to_explain=image, heatmap_a=heatmap_a, heatmap_b=heatmap_b)

    cv2.destroyAllWindows()