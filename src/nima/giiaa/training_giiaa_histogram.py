"""
Training script for distribution-based GIIAA, based on the NIMA paper from Google.
"""

import os
import pandas as pd
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from src.nima.giiaa.base_module_giiaa import BaseModuleGIIAA

LIGHT_AND_COLOR_CSV = "datasets/eva/metadata/lightAndColor.csv"

LOG_PATH = "docs/nima_fine_tune_logs"
MODELS_PATH = "models/"
MODEL_NAME_TAG = 'nima_fine_tune_'
BASE_MODEL_NAME = "InceptionResNetV2"

N_CLASSES = 10
BATCH_SIZE = 32
DROPOUT_RATE = 0.75
USE_MULTIPROCESSING = False
N_WORKERS = 1

EPOCHS_DENSE = 9
LEARNING_RATE_DENSE = 0.1
DECAY_DENSE = 0

EPOCHS_ALL = 5
LEARNING_RATE_ALL = 0.00003
DECAY_ALL = 0.000023

WEIGHTS = 'models/nima_loss-0.078.hdf5'

if __name__ == "__main__":

    nima = BaseModuleGIIAA(custom_weights=WEIGHTS, base_model_name=BASE_MODEL_NAME, n_classes=N_CLASSES, learning_rate=LEARNING_RATE_DENSE, decay_rate=DECAY_DENSE, dropout_rate=DROPOUT_RATE)
    nima.build()

    dataframe = pd.read_csv(LIGHT_AND_COLOR_CSV, converters={'transformed_score': eval})

    data_generator = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2
    )

    train_generator = data_generator.flow_from_dataframe(
        dataframe=dataframe,
        x_col='image_path',
        y_col=['transformed_score'],
        class_mode='multi_output',
        target_size=(224, 224),
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        subset='training'
    )

    validation_generator = data_generator.flow_from_dataframe(
        dataframe=dataframe,
        x_col='image_path',
        y_col=['transformed_score'],
        class_mode='multi_output',
        target_size=(224, 224),
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        subset='validation',
    )

    tensorboard = TensorBoard(
        log_dir=LOG_PATH, update_freq='batch'
    )

    model_save_name = (MODEL_NAME_TAG + BASE_MODEL_NAME.lower() + '_{loss:.3f}.hdf5')
    model_file_path = os.path.join(MODELS_PATH, model_save_name)
    model_checkpointer = ModelCheckpoint(
        filepath=model_file_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
    )

    for layer in nima.base_model.layers[:-1]:
        layer.trainable = False

    nima.compile()
    nima.nima_model_oneclass.summary()

    steps = train_generator.samples // train_generator.batch_size
    # steps = 50

    nima.nima_model_oneclass.fit_generator(
        generator=train_generator,
        steps_per_epoch=steps,
        validation_data=validation_generator,
        validation_steps=steps,
        epochs=EPOCHS_DENSE,
        use_multiprocessing=USE_MULTIPROCESSING,
        workers=N_WORKERS,
        verbose=1,
        max_queue_size=30,
        callbacks=[tensorboard, model_checkpointer]
    )

    for layer in nima.base_model.layers:
        layer.trainable = True

    nima.compile()

    nima.nima_model_oneclass.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        epochs=EPOCHS_DENSE + EPOCHS_ALL,
        initial_epoch=EPOCHS_DENSE,
        use_multiprocessing=USE_MULTIPROCESSING,
        workers=N_WORKERS,
        verbose=1,
        max_queue_size=30,
        callbacks=[tensorboard, model_checkpointer],
    )

    K.clear_session()




