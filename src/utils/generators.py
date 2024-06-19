"""
The huge amount of data requires us to use data generators to feed images into the NN.
This can be handled with the keras' ImageDataGenerator class alone, but to generate
pairs of images for the GCIAA-categories and GCIAA-distortions, we need to create
our own Generator classes.
"""

import random
import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from jax import numpy as jnp

SEED = 21


class SiameseGeneratorCategories:

    def __init__(self, generator, dataframe, subset=None, target_size=(224, 224), color_mode='rgb', shuffle=True, batch_size=96):

        self.generator_a = generator.flow_from_dataframe(
            dataframe=dataframe,
            target_size=target_size,
            color_mode=color_mode,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=SEED,
            x_col='id_a',
            y_col='label',
            class_mode='raw',
            subset=subset
        )

        self.generator_b = generator.flow_from_dataframe(
            dataframe=dataframe,
            target_size=target_size,
            color_mode=color_mode,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=SEED,
            x_col='id_b',
            y_col='label',
            class_mode='raw',
            subset=subset
        )

        self.samples_per_epoch = self.generator_a.samples
        self.batch_size = batch_size

        self.distortion_blob = dict(radius=30)

    def get_pairwise_flow_from_dataframe(self):

        while True:
            (image_batch_a, label_batch) = self.generator_a.next()
            (image_batch_b, _) = self.generator_b.next()

            yield [image_batch_a, image_batch_b], [label_batch]


class SiameseGeneratorDistortions:
    def __init__(self, generator, distortion_generators, dataframe, subset=None, target_size=(224, 224), color_mode='rgb', shuffle=True, batch_size=96):

        self.flow_original = generator.flow_from_dataframe(
            dataframe=dataframe,
            target_size=target_size,
            color_mode=color_mode,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=SEED,
            x_col='id',
            y_col='label',
            class_mode='raw',
            subset=subset
        )

        self.flow_distortions = []
        for generator in distortion_generators:
            self.flow_distortions.append(generator.flow_from_dataframe(
                dataframe=dataframe,
                target_size=target_size,
                color_mode=color_mode,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=SEED,
                x_col='id',
                y_col='label',
                class_mode='raw',
                subset=subset
            ))

        self.samples_per_epoch = self.flow_original.samples
        self.batch_size = batch_size
        self.distortion_type = 0

    @staticmethod
    def apply_black_and_white(image):
        image = np.array(image)
        distorted_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        distorted_image = cv2.cvtColor(distorted_image, cv2.COLOR_GRAY2BGR)
        return distorted_image

    @staticmethod
    def apply_blur(image):
        image = np.array(image)
        distorted_image = cv2.blur(image, (30, 30))
        return distorted_image

    @staticmethod
    def apply_blob_overlay(image):
        distorted_image = np.array(image)

        radius_x = random.randrange(50, 112)
        radius_y = random.randrange(50, 112)
        center_x = random.randrange(radius_x, distorted_image.shape[0] - radius_x)
        center_y = random.randrange(radius_y, distorted_image.shape[1] - radius_y)

        distorted_image[center_x - radius_x // 2:center_x + radius_x // 2, center_y - radius_y // 2:center_y + radius_y // 2] = \
            [random.random() * 255.0, random.random() * 255.0, random.random() * 255.0]
        return distorted_image

    def get_pairwise_flow_from_dataframe(self):
        while True:

            (image_batch_a, label_batch) = self.flow_original.next()
            (image_batch_b, _) = self.flow_distortions[self.distortion_type].next()

            seed = [float(random.random() < 0.5) for _ in range(len(label_batch))]

            shuffled_image_batch_a = []
            shuffled_image_batch_b = []
            shuffled_label_batch = []

            for i in range(len(seed)):
                (image_a, image_b) = \
                    (image_batch_a[i, :], image_batch_b[i, :]) if seed[i] == 0.0 else \
                    (image_batch_b[i, :], image_batch_a[i, :])

                shuffled_image_batch_a.append(image_a)
                shuffled_image_batch_b.append(image_b)
                shuffled_label_batch.append(seed[i])

            shuffled_image_batch_a = np.asarray(shuffled_image_batch_a)
            shuffled_image_batch_b = np.asarray(shuffled_image_batch_b)
            shuffled_label_batch = np.asarray(shuffled_label_batch)

            yield [shuffled_image_batch_a, shuffled_image_batch_b], [shuffled_label_batch]

            self.distortion_type = self.distortion_type + 1 if self.distortion_type + 1 < len(self.flow_distortions) else 0


class ImageDataLoader:
    def __init__(self, data_folder, csv_file, model_arch=None, batch_size=32):
        self.data_folder = data_folder
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.model_arch = model_arch

        # Load CSV file with target scores
        self.df = pd.read_csv(csv_file)

        # Split dataset into train and test sets
        self.train_df, self.test_df = train_test_split(self.df, test_size=0.2, random_state=42)
        self.num_train_samples = len(self.train_df)

    def __len__(self):
        return len(self.train_df) // self.batch_size

    def __iter__(self):
        num_batches = len(self)        
        for idx in range(num_batches):
            batch_indices = np.arange(idx * self.batch_size, (idx + 1) * self.batch_size)
            batch_data = self.train_df.iloc[batch_indices]
            # Assuming you have a function load_image_data to load image data from file paths
            inputs = [self.model_arch.prepare_image(os.path.join(self.data_folder, "{}.jpg".format(filename)), self.model_arch.pp_config, should_resize=True) for filename in batch_data['image_id']]
            inputs = jnp.stack(inputs)
            targets = batch_data['transformed_score'].values
            yield inputs, targets