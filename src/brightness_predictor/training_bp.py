import os
import pandas as pd
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator

from src.brightness_predictor.base_module_bp import BaseModuleBP

BP_DATAFRAME_PATH = "datasets/eva/metadata/dataframe_bp.csv"

LOG_PATH = "docs/training_history/"
MODELS_PATH = "models/"
MODEL_NAME_TAG = 'brightness_predictor'
BASE_MODEL_NAME = "InceptionResNetV2"

N_CLASSES = 1
BATCH_SIZE = 96
DROPOUT_RATE = 0.75
USE_MULTIPROCESSING = False
N_WORKERS = 1

EPOCHS_DENSE = 9
LEARNING_RATE_DENSE = 0.001
DECAY_DENSE = 0

EPOCHS_ALL = 5
LEARNING_RATE_ALL = 0.00003
DECAY_ALL = 0.000023

if __name__ == "__main__":
    bp = BaseModuleBP()
    bp.build()

    dataframe = pd.read_csv(BP_DATAFRAME_PATH, converters={'label': eval})

    data_generator = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2
    )

    train_generator = data_generator.flow_from_dataframe(
        dataframe=dataframe,
        x_col='id',
        y_col=['label'],
        class_mode='multi_output',
        target_size=(224, 224),
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        subset='training'
    )

    validation_generator = data_generator.flow_from_dataframe(
        dataframe=dataframe,
        x_col='id',
        y_col=['label'],
        class_mode='multi_output',
        target_size=(224, 224),
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        subset='validation',
    )

    tensorboard = TensorBoard(
        log_dir=LOG_PATH, update_freq='batch'
    )

    model_save_name = (MODEL_NAME_TAG + BASE_MODEL_NAME.lower() + '_{val_loss:.3f}.hdf5')
    model_file_path = os.path.join(MODELS_PATH, model_save_name)
    model_checkpointer = ModelCheckpoint(
        filepath=model_file_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
    )

    for layer in bp.base_model.layers:
        layer.trainable = False

    bp.compile()
    bp.bp_model.summary()

    # steps = train_generator.samples // train_generator.batch_size
    steps = 50

    bp.bp_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=steps,
        validation_data=validation_generator,
        validation_steps=steps,
        epochs=1,
        use_multiprocessing=USE_MULTIPROCESSING,
        workers=N_WORKERS,
        verbose=1,
        max_queue_size=30,
        callbacks=[tensorboard, model_checkpointer]
    )

    for layer in bp.base_model.layers:
        layer.trainable = True

    bp.compile()
    bp.bp_model.summary()

    bp.bp_model.fit_generator(
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