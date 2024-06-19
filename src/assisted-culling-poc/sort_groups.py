"""
Using distribution-based GIIAA to make inference on a few random samples from the subset of images.
"""

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

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

from src.musiq.siamese_transformer_base import SiameseTransformerBase


if __name__ == "__main__":
    # run_and_save_multiple_images(LocalConfig.MODEL, LocalConfig.INPUT_PATH, LocalConfig.OUTPUT_PATH)
    musiq = SiameseTransformerBase()
    score = musiq.run_model_multiple_images("datasets/evaluation-groups/model")
    print(score)
    # trichy puppy model-convertToJPG jaipur taj goat monastery everest plant sunset
