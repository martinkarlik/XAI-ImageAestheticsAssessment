"""
Evaluation of distortion-trained NIMA model.
Each distortion is evaluated on 12 775 distortion-augmented pairs generated from AVA dataset.
Distortions are evaluated individually, the performance of the distortion-trained NIMA model
is compared with the baseline NIMA model with NIMA model as the image encoder.

UNDEREXPOSURE   Baseline    |   Distortion trained NIMA
Loss:           0.4432      |   0.1060
Accuracy:       0.7948      |   0.9580

OVEREXPOSURE    Baseline    |   Distortion trained NIMA
Loss:           0.4512      |   0.6056
Accuracy:       0.7531      |   0.3561

ROTATION        Baseline    |   Distortion trained NIMA
Loss:           0.3825      |   0.0566
Accuracy:       0.9742      |   0.9841

SHEARING        Baseline    |   Distortion trained NIMA
Loss:           0.4282      |   0.0553
Accuracy:       0.8120      |   0.9822

ZOOMING         Baseline    |   Distortion trained NIMA
Loss:           0.3726      |   0.0531
Accuracy:       0.9508      |   0.9850

BLUR            Baseline    |   Distortion trained NIMA
Loss:           0.3749      |   0.1900
Accuracy:       0.9265      |   0.8462

BLOB OVERLAY    Baseline    |   Distortion trained NIMA
Loss:           0.4951      |   0.3202
Accuracy:       0.7190      |   0.9036

----------------------------------------------------------

ALL DISTORTIONS Baseline    |   Distortion trained NIMA
Loss:           0.4211      |   0.1981
Accuracy:       0.8472      |   0.8757
"""

import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

from src.utils.generators import *
from src.nima.gciaa.base_module_gciaa import BaseModuleGCIAA

GIIAA_MODEL = "models/nima_loss-0.078.hdf5"
GCIAA_DISTORTIONS_MODEL = "models/nima_dist_acc-0.865.hdf5"

AVA_DATAFRAME_TEST_PATH = "datasets/ava/metadata/gciaa_metadata/dataframe_AVA_gciaa-dist_test.csv"


DISTORTION_GENERATORS = [
    ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2, brightness_range=[0.2, 0.75]),  # Underexposure
    ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2, brightness_range=[1.5, 5.0]),   # Overexposure
    ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2, rotation_range=90.0),           # Rotation
    ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2, shear_range=90.0),              # Shearing
    ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2, zoom_range=0.5),                # Zooming
    ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2,
                       preprocessing_function=SiameseGeneratorDistortions.apply_blur),          # Blur
    ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2,
                       preprocessing_function=SiameseGeneratorDistortions.apply_blob_overlay)   # Blob overlay
]


if __name__ == "__main__":

    base = BaseModuleGCIAA(custom_weights=GIIAA_MODEL, load_weights_as='GIIAA')
    base.build()
    base.compile()

    base.siamese_model.load_weights(GCIAA_DISTORTIONS_MODEL)
    base.siamese_model.summary()

    dataframe = pd.read_csv(AVA_DATAFRAME_TEST_PATH, converters={'label': eval})

    data_generator = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2
    )

    distortion_generators = [
        DISTORTION_GENERATORS[6]
    ]

    test_generator = SiameseGeneratorDistortions(
        generator=data_generator,
        distortion_generators=distortion_generators,
        dataframe=dataframe)

    accuracy = base.siamese_model.evaluate_generator(
        generator=test_generator.get_pairwise_flow_from_dataframe(),
        steps=test_generator.samples_per_epoch / test_generator.batch_size,
        verbose=1
    )

