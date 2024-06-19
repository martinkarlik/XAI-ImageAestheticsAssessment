"""
Using distribution-based GIIAA on custom images.
"""

import os

import cv2
import numpy as np
import onnxruntime

from src.nima.giiaa.base_module_giiaa import BaseModuleGIIAA

H5_MODEL_PATH = "../../../models/giiaa-hist_200k_base-inceptionresnetv2_loss-0.078.hdf5"
ONNX_MODEL_PATH = "../../models/giiaa_model.onnx"

CUSTOM_IMAGES_PATH = "../../data/custom/dataset/"
SORTED_CLUSTERS_PATH = "../../data/custom/sorted_clusters/"
SORTED_ALL_PATH = "../../data/custom/sorted_all/"


def get_weighted_mean(distribution):
    mean = 0.0
    for i in range(0, len(distribution)):
        mean += distribution[i] * (i + 1)
    return mean


if __name__ == "__main__":

    nima = BaseModuleGIIAA(custom_weights=H5_MODEL_PATH)
    nima.build()
    nima.compile()

    session = onnxruntime.InferenceSession(ONNX_MODEL_PATH)

    annotated_images = {}

    for filename in os.listdir(CUSTOM_IMAGES_PATH):
        if not filename.lower().endswith('.jpg'):
            continue

        file = os.path.join(CUSTOM_IMAGES_PATH, filename)
        image = cv2.resize(cv2.imread(file), (224, 224)) / 255.0
        image = np.asarray(image)[np.newaxis, ...].astype('float32')

        print("H5")
        h5_prediction_histogram = nima.nima_model.predict(image)[0]
        h5_prediction_weighted_mean = get_weighted_mean(h5_prediction_histogram)
        print(h5_prediction_weighted_mean)

        inname = [input.name for input in session.get_inputs()]
        outname = [output.name for output in session.get_outputs()]
        result = session.run(outname, {inname[0]: image})

        print("ONNX")
        onnx_prediction_histogram = result[0][0]
        onnx_prediction_weighted_mean = get_weighted_mean(onnx_prediction_histogram)
        print(onnx_prediction_weighted_mean)

        annotated_images[filename] = onnx_prediction_weighted_mean

    annotated_images = sorted(annotated_images.items(), key=lambda item: item[1])

    # Clear destination folders
    for cluster_dir in os.listdir(SORTED_CLUSTERS_PATH):
        for filepath in os.listdir(SORTED_CLUSTERS_PATH + cluster_dir):
            if not filepath.lower().endswith('.jpg'):
                continue
            os.remove(SORTED_CLUSTERS_PATH + cluster_dir + "/" + filepath)

    for filepath in os.listdir(SORTED_ALL_PATH):
        if not filepath.lower().endswith('.jpg'):
            continue
        os.remove(SORTED_ALL_PATH + filepath)

    # Sort within clusters
    for i in range(0, len(annotated_images)):
        image_filename = annotated_images[i][0]
        image_score = annotated_images[i][1]

        src_image_path = CUSTOM_IMAGES_PATH + image_filename
        clustername = image_filename.split('_')[0] + "/"
        dest_dir_path = SORTED_CLUSTERS_PATH + clustername

        if not os.path.exists(dest_dir_path):
            os.makedirs(dest_dir_path)

        image_num = len(os.listdir(dest_dir_path))

        dest_filename = "img{}-{:.2f}.jpg".format(str(image_num).zfill(3), image_score)
        dest_image_path = dest_dir_path + dest_filename

        cv2.imwrite(dest_image_path, cv2.imread(src_image_path))

    # Sort all
    for i in range(0, len(annotated_images)):
        image_filename = annotated_images[i][0]
        image_score = annotated_images[i][1]

        image_num = len(os.listdir(SORTED_ALL_PATH))
        dest_filename = "img{}-{:.2f}.jpg".format(str(image_num).zfill(3), image_score)

        src_image_name = CUSTOM_IMAGES_PATH + image_filename
        dest_image_name = SORTED_ALL_PATH + dest_filename

        cv2.imwrite(dest_image_name, cv2.imread(src_image_name))



