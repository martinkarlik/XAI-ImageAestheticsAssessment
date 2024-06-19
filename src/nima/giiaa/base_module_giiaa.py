"""
The implementation of this script follows the paper NIMA from Google,
apart from implementing random crop to preprocess the image inputs.
Our implementation uses InceptionResNetV2 as the base model.
"""

import importlib

import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay

import numpy as np


def earth_movers_distance(y_true, y_predicted):

    cdf_true = K.cumsum(y_true, axis=-1)
    cdf_predicted = K.cumsum(y_predicted, axis=-1)
    emd = K.sqrt(K.mean(K.square(cdf_true - cdf_predicted), axis=-1))

    return K.mean(emd)

def squared_mean_error_loss(y_true, y_predicted):
    squared_difference = K.square(y_true - y_predicted)
    mean_squared_error = K.mean(squared_difference)

    # tf.print("Debugging the loss function")
    # tf.print(y_true)
    # tf.print(y_predicted)
    # tf.print(squared_difference)
    # tf.print(mean_squared_error)

    return mean_squared_error

def histogram_mean_layer(vectors):
    rank_vector = K.constant(np.array([i for i in range(1, 11)]).reshape(1, 10))

    means = K.dot(rank_vector, K.transpose(vectors))

    return means


class BaseModuleGIIAA:

    def __init__(self, custom_weights=None, weights='imagenet', base_model_name="InceptionResNetV2", n_classes=10,
                 learning_rate=0.001, dropout_rate=0, decay_rate=0.9, decay_steps=10000):

        self.base_model_name = base_model_name
        self.n_classes = n_classes
        self.weights = weights
        self.custom_weights = custom_weights

        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate

        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

        if self.base_model_name == "InceptionResNetV2":
            self.base_module = importlib.import_module('tensorflow.keras.applications.inception_resnet_v2')
        elif self.base_model_name == "InceptionV3":
            self.base_module = importlib.import_module('tensorflow.keras.applications.inception_v3')
        else:
            self.base_module = importlib.import_module('tensorflow.keras.applications.' + self.base_model_name.lower())

        self.base_model = None
        self.nima_model = None
        self.nima_model_unpacked = None
        self.nima_model_oneclass = None

    def build(self):

        imagenet_cnn = getattr(self.base_module, self.base_model_name)

        # Replace last layer with Dropout and Dense (virtually turn classification into multi-output regression).
        self.base_model = imagenet_cnn(input_shape=(224, 224, 3), weights=self.weights, include_top=False, pooling='avg')

        x = Dropout(self.dropout_rate)(self.base_model.output)
        x = Dense(units=self.n_classes, activation='softmax')(x)

        self.nima_model = Model(self.base_model.inputs, x)
        if self.custom_weights:
            self.nima_model.load_weights(self.custom_weights)

            self.nima_model_unpacked = Model(self.nima_model.inputs, self.nima_model.layers[-3].output)
            self.nima_model_unpacked.set_weights(self.nima_model.get_weights()[:-2])

            output_layer = Dense(units=1, activation='softmax')(self.nima_model.output)
            self.nima_model_oneclass = Model(inputs=self.nima_model.inputs, outputs=output_layer)
            new_layer_weights = [np.random.randn(*w.shape) for w in self.nima_model_oneclass.layers[-1].get_weights()]
            weights_to_transfer = self.nima_model.get_weights() + new_layer_weights
            self.nima_model_oneclass.set_weights(weights_to_transfer)

    def compile(self):
        lr_schedule = ExponentialDecay(
            self.learning_rate, decay_steps=self.decay_steps, decay_rate=self.decay_rate, staircase=True
        )

        self.nima_model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=earth_movers_distance)
        self.nima_model_oneclass.compile(optimizer=Adam(learning_rate=lr_schedule), loss=squared_mean_error_loss)

    def predict(self, image):
        self.nima_model.predict(image)

    def predict_base(self, image):
        self.base_model.predict(image)

    def predict_oneclass(self, image):
        self.nima_model_oneclass.predict(image)

    def preprocess(self):
        return self.base_module.preprocess_input
