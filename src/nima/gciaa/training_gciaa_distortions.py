"""
Training script for distribution-based GCIAA.
We are using 7 types of distortions: underexposure, overexposure, zooming, shearing, rotation, blur, blob overlay.
To generate the image pairs, we are using a custom generator defined in utils/generators.py file.
"""

import os

import pandas as pd
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator

from src.nima.gciaa.base_module_gciaa import BaseModuleGCIAA
from src.utils.generators import *

AVA_DATAFRAME_PATH = "datasets/ava/metadata/gciaa_metadata/dataframe_AVA_gciaa-dist_train.csv"

GIIAA_MODEL = "models/nima_loss-0.078.hdf5"

MODELS_PATH = "models/"
MODEL_NAME_TAG = 'blur_v3'


LOG_PATH = "docs/logs/blur_logs_v2"
BASE_MODEL_NAME = "nima"
BATCH_SIZE = 16
DROPOUT_RATE = 0.75
USE_MULTIPROCESSING = False
N_WORKERS = 1

EPOCHS = 10


if __name__ == "__main__":

    tensorboard = TensorBoard(
        log_dir=LOG_PATH, update_freq='batch'
    )

    model_save_name = (MODEL_NAME_TAG + '_{accuracy:.3f}.hdf5')
    model_file_path = os.path.join(MODELS_PATH, model_save_name)
    model_checkpointer = ModelCheckpoint(
        filepath=model_file_path,
        monitor='accuracy',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )

    base = BaseModuleGCIAA(custom_weights=GIIAA_MODEL, load_weights_as='GIIAA')
    base.build()
    base.compile()

    # Training the GCIAA model with artificially created pairs of images.
    dataframe = pd.read_csv(AVA_DATAFRAME_PATH, converters={'label': eval})

    data_generator = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2
    )

    # distortion_generators = [
    #     ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2, brightness_range=[0.2, 0.75]),
    #     ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2, brightness_range=[1.5, 5.0]),
    #     ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2, rotation_range=90.0),
    #     ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2, shear_range=90.0),
    #     ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2, zoom_range=0.5),
    #     ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2, preprocessing_function=SiameseGeneratorDistortions.apply_blur),
    #     ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2, preprocessing_function=SiameseGeneratorDistortions.apply_blob_overlay)
    # ]

    distortion_generators = [
        ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2, preprocessing_function=SiameseGeneratorDistortions.apply_blur)
    ]

    train_generator = SiameseGeneratorDistortions(
        generator=data_generator,
        distortion_generators=distortion_generators,
        dataframe=dataframe,
        batch_size=BATCH_SIZE,
        subset='training')

    validation_generator = SiameseGeneratorDistortions(
        generator=data_generator,
        distortion_generators=distortion_generators,
        dataframe=dataframe,
        batch_size=BATCH_SIZE,
        subset='validation')

    base.siamese_model.fit(
        train_generator.get_pairwise_flow_from_dataframe(),
        steps_per_epoch=train_generator.samples_per_epoch // train_generator.batch_size // 10,
        validation_data=validation_generator.get_pairwise_flow_from_dataframe(),
        validation_steps=validation_generator.samples_per_epoch // validation_generator.batch_size // 10,
        epochs=EPOCHS,
        use_multiprocessing=USE_MULTIPROCESSING,
        workers=N_WORKERS,
        verbose=1,
        max_queue_size=30,
        callbacks=[tensorboard, model_checkpointer]
    )

    K.clear_session()
