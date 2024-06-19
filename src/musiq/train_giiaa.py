# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run inference on a single image with a MUSIQ checkpoint."""

import collections
import io
import os
from typing import Dict, Sequence, Text, TypeVar

from absl import app
from absl import flags
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import tensorflow.compat.v1 as tf

import src.musiq.multiscale_transformer as model_mod
import src.musiq.preprocessing as pp_lib

from src.musiq.siamese_transformer_base import SiameseTransformerBase
from src.utils.generators import ImageDataLoader
WEIGHTS = 'models/spaq_ckpt.npz'

IMAGE_DATASET = 'datasets/eva/images'
LABELS = 'datasets/eva/metadata/lightAndColor.csv'


if __name__ == '__main__':

    musiq = SiameseTransformerBase(ckpt_path=WEIGHTS)

    image_data_loader = ImageDataLoader(data_folder=IMAGE_DATASET, model_arch=musiq, csv_file=LABELS)

    musiq.train_giiaa(data_loader=image_data_loader)