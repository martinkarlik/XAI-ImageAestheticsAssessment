
import src.musiq.multiscale_transformer as model_mod
import src.musiq.preprocessing as pp_lib
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
from flax import nn, optim
import tensorflow.compat.v1 as tf
import cv2


# Set to True when using full-size single-scale checkpoints.
_SINGLE_SCALE = False

# Image preprocessing config.
_PP_CONFIG = {
    'patch_size': 32,
    'patch_stride': 32,
    'hse_grid_size': 10,
    'longer_side_lengths': [] if _SINGLE_SCALE else [224, 384],
    # -1 means using all the patches from the full-size image.
    'max_seq_len_from_original_res': -1,
}

# Model backbone config.
_MODEL_CONFIG = {
    'hidden_size': 384,
    'representation_size': None,
    'resnet_emb': {
        'num_layers': 5
    },
    'transformer': {
        'attention_dropout_rate': 0,
        'dropout_rate': 0,
        'mlp_dim': 1152,
        'num_heads': 6,
        'num_layers': 14,
        'num_scales': 1 if _SINGLE_SCALE else 3,
        'spatial_pos_grid_size': 10,
        'use_scale_emb': True,
        'use_sinusoid_pos_emb': False,
    }
}

T = TypeVar('T')  # Declare type variable

class LocalConfig(enumerate):
    NUM_CLASSES = 1
    TRAIN = True
    CKPT_PATH = 'models/koniq_ckpt.npz'



class SiameseTransformerBase:

    def __init__(self, ckpt_path=LocalConfig.CKPT_PATH):

        self.model_config, self.pp_config, self.params = self.get_params_and_config(ckpt_path=ckpt_path)
        self.image_encoder_model = model_mod.Model.partial(num_classes=LocalConfig.NUM_CLASSES, train=LocalConfig.TRAIN, **self.model_config)
        self.unpacked_model = model_mod.UnpackedModel.partial(num_classes=LocalConfig.NUM_CLASSES, train=LocalConfig.TRAIN, **self.model_config)
        # self.optimizer = optim.Adam(learning_rate=1e-4).create(self.image_encoder_model.init_by_shape(jax.random.PRNGKey(0), [(1, 224, 224, 3)]))

        # self.siamese_model = model_mod.SiameseModel(self.image_encoder_model)
            
    def run_siamese_model(self, image_path1, image_path2):

        image1 = self.prepare_image(image_path1)
        image2 = self.prepare_image(image_path2)

        return self.siamese_model(self.params, image1, image2)
    
    def run_model_for_lime(self, image):
        try:
            image = self.prepare_image_from_array(image)
            logits = self.image_encoder_model.call(self.params, image)
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
        preds = logits
        if LocalConfig.NUM_CLASSES > 1:
            preds = jax.nn.softmax(logits)
        score_values = jnp.arange(1, LocalConfig.NUM_CLASSES + 1, dtype=np.float32)
        score = jnp.sum(preds * score_values, axis=-1)[0]

        return score

    def run_model_single_image(self, image_path):
        
        if not (image_path.endswith(".jpg") or image_path.endswith(".jpeg") or image_path.endswith(".png")):
            return None

        try:
            image = self.prepare_image(image_path)
            logits = self.image_encoder_model.call(self.params, image)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
        preds = logits
        if LocalConfig.NUM_CLASSES > 1:
            preds = jax.nn.softmax(logits)
        score_values = jnp.arange(1, LocalConfig.NUM_CLASSES + 1, dtype=np.float32)
        score = jnp.sum(preds * score_values, axis=-1)[0]

        return score
    
    def run_model_multiple_images(self, folder_path):

        scores = {}

        for filename in os.listdir(folder_path):
            scores[filename] = self.run_model_single_image(os.path.join(folder_path, filename))

        return scores
    
    def run_model_unpacked(self, image_path):
        
        if not (image_path.endswith(".jpg") or image_path.endswith(".jpeg") or image_path.endswith(".png")):
            return None

        try:
            image = self.prepare_image(image_path)
            logits = self.unpacked_model.call(self.params, image)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

        return logits
    

    def prepare_image_from_array(self, image_array):
        # Ensure the image is a numpy array of the right shape
        assert isinstance(image_array, np.ndarray), "Input must be a numpy array"
        assert image_array.shape == (224, 224, 3), "Input array must be of shape (224, 224, 3)"

        # Convert the numpy array to uint8 type
        image_array = image_array.astype(np.uint8)

        # Encode the numpy array as a JPEG image string
        encoded_str = tf.image.encode_jpeg(tf.convert_to_tensor(image_array)).numpy()

        data = dict(image=tf.constant(encoded_str))
        pp_fn = pp_lib.get_preprocess_fn(**self.pp_config)
        data = pp_fn(data)
        image = data['image']
        
        # Shape (1, length, dim)
        image = tf.expand_dims(image, axis=0)
        image = image.numpy()
        return image
    
    def prepare_image(self, image_path):

        with tf.compat.v1.gfile.FastGFile(image_path, 'rb') as f:
            encoded_str = f.read()

        data = dict(image=tf.constant(encoded_str))
        pp_fn = pp_lib.get_preprocess_fn(**self.pp_config)
        data = pp_fn(data)
        image = data['image']
        # Shape (1, length, dim)
        image = tf.expand_dims(image, axis=0)
        image = image.numpy()
        return image

    def recover_tree(self, keys, values):
        tree = {}
        sub_trees = collections.defaultdict(list)
        for k, v in zip(keys, values):
            if '/' not in k:
                tree[k] = v
            else:
                k_left, k_right = k.split('/', 1)
                sub_trees[k_left].append((k_right, v))
        for k, kv_pairs in sub_trees.items():
            k_subtree, v_subtree = zip(*kv_pairs)
            tree[k] = self.recover_tree(k_subtree, v_subtree)
        return tree

    def get_params_and_config(self, ckpt_path):
        """Returns (model config, preprocessing config, model params from ckpt)."""
        model_config = ml_collections.ConfigDict(_MODEL_CONFIG)
        pp_config = ml_collections.ConfigDict(_PP_CONFIG)

        with tf.compat.v1.gfile.FastGFile(ckpt_path, 'rb') as f:
            data = f.read()
        values = np.load(io.BytesIO(data))
        params = self.recover_tree(*zip(*values.items()))
        params = params['opt']['target']
        if not model_config.representation_size:
            params['pre_logits'] = {}

        return model_config, pp_config, params
    
    def compute_loss(logits, targets):
    # Assuming logits are the predicted quality scores and targets are the ground truth scores
    # Compute the Mean Squared Error (MSE) loss
        loss = jnp.mean((logits - targets) ** 2)
        return loss

    def train_giiaa(self, data_loader, num_epochs=1):

        def train_step(params, optimizer, batch):
            # Define your loss function
            def loss_fn(params, batch):
                inputs, targets = batch

                current_image = self.prepare_image()
                logits = self.image_encoder_model.apply({'params': params}, inputs)
                loss = self.compute_loss(logits, targets)  # You need to define compute_loss function
                return loss

            # Compute gradients using jax.grad
            grads = jax.grad(loss_fn)(params, batch)
            
            # Update the optimizer state
            updates, optimizer = optimizer.update(grads)
            
            # Update the parameters using optax.apply_updates
            new_params = optax.apply_updates(params, updates)
            
            # Compute the loss for logging
            loss = loss_fn(new_params, batch)
            
            return new_params, optimizer, loss

        for epoch in range(num_epochs):
            for batch in data_loader:
                self.params, self.optimizer, loss = train_step(self.params, self.optimizer, batch)
                print(f"Epoch {epoch+1}, Loss: {loss}")

            # Save checkpoints after each epoch if needed
            self.save_checkpoint(f"models/ckpt_epoch_{epoch+1}.npz")