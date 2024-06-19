"""
This script provides the base module for the Comparative IAA model.
The pipeline of the network is:
    two images -> image encoder -> image encodings -> mapping to a binary output -> index of the better image.
As the image encoder, we are using the trained GIIAA model. The inputs-image encoder relationship follows
the typical Siamese model. However, the mapping function is custom: The image encoding is the GIIAA histogram output,
so we compute the mean value of this histogram, subtract the two mean values of the two images' encodings' and pass it
into a sigmoid function. The output can than be rounded to get the index of the preferred image.
"""

import importlib
from time import time_ns

import numpy as np
from keras import Input
from keras import backend as K
from keras.layers import Dense, Dropout, Lambda
from keras.models import Model
from keras.optimizers import RMSprop
from keras.optimizers.schedules import ExponentialDecay


def contrastive_loss(y_true, y_predicted):
    return K.mean(y_true * K.maximum(1 - y_predicted, 0) + (1 - y_true) * y_predicted)


def accuracy(y_true, y_predicted):
    return K.mean(K.equal(y_true, K.cast(y_predicted > 0.5, y_true.dtype)))


def mapped_comparison_layer(vectors):
    (features_a, features_b) = vectors
    rank_vector = K.constant(np.array([i for i in range(1, 11)]).reshape(1, 10))

    means_a = K.dot(rank_vector, K.transpose(features_a))
    means_b = K.dot(rank_vector, K.transpose(features_b))
    distance = K.transpose(means_b - means_a)

    mapped_distance = 1.0 / (1.0 + K.exp(-distance))

    # K.print_tensor(features_a)
    # K.print_tensor(features_b)
    # K.print_tensor(mapped_distance)

    return mapped_distance


class BaseModuleGCIAA:

    def __init__(self, custom_weights=None, load_weights_as='GCIAA', base_model_name="InceptionResNetV2", n_classes_base=1, loss=contrastive_loss, learning_rate=0.00001,
                 dropout_rate=0, decay_rate=0.9, decay_steps=10000):

        self.weights = custom_weights
        self.load_weights_as = load_weights_as

        self.base_model_name = base_model_name
        self.n_classes_base = n_classes_base
        self.loss = loss

        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate

        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

        if self.base_model_name == 'InceptionResNetV2':
            self.base_module = importlib.import_module('tensorflow.keras.applications.inception_resnet_v2')
        else:
            raise Exception("Trying to use unknown model: {}.".format(self.base_model_name))

        self.image_encoder_model = None
        self.image_encoder_model_unpacked = None
        self.siamese_model = None
        self.siamese_model_unpacked = None

    def build(self):

        imagenet_cnn = getattr(self.base_module, self.base_model_name)

        # Remove the last layer from InceptionResnetV2 (turn classification into a base for siamese network).
        imagenet_model = imagenet_cnn(input_shape=(224, 224, 3), weights=None, include_top=False, pooling='avg')

        x = Dropout(self.dropout_rate)(imagenet_model.output)
        x = Dense(units=10, activation='softmax')(x)

        # Set image encode model to the trained GIIAA model.
        self.image_encoder_model = Model(imagenet_model.inputs, x)
        self.image_encoder_model_unpacked = Model(imagenet_model.inputs, imagenet_model.output)
        if self.load_weights_as == 'GIIAA':
            self.image_encoder_model.load_weights(self.weights)
            self.image_encoder_model_unpacked.set_weights(self.image_encoder_model.get_weights()[:-2])

        # Build the siamese model.
        image_a = Input(shape=(224, 224, 3), dtype='float32')
        image_b = Input(shape=(224, 224, 3), dtype='float32')
        encoding_a = self.image_encoder_model(image_a)
        encoding_b = self.image_encoder_model(image_b)

        x = Lambda(mapped_comparison_layer, name="mapped_comparison_layer", trainable=False, output_shape=(1,))([encoding_a, encoding_b])

        self.siamese_model = Model(inputs=[image_a, image_b], outputs=x)
        if self.load_weights_as == 'GCIAA':
            self.siamese_model.load_weights(self.weights)

            self.siamese_model_unpacked = Model(imagenet_model.inputs, imagenet_model.output)
            self.siamese_model_unpacked.set_weights(self.siamese_model.layers[2].get_weights()[:-2])

            self.image_encoder_model.set_weights(self.siamese_model.layers[2].get_weights())

            self.siamese_model_unpacked.summary()

    def compile(self):
        lr_schedule = ExponentialDecay(
            self.learning_rate, decay_steps=self.decay_steps, decay_rate=self.decay_rate, staircase=True
        )
        self.siamese_model.compile(optimizer=RMSprop(learning_rate=lr_schedule), loss=self.loss, metrics=[accuracy])

    def predict(self, inputs):
        prediction = self.siamese_model.predict([inputs[0], inputs[1]])
        return prediction
    
    def predict_image_encoder(self, input):
        prediction = self.image_encoder_model.predict(input)
        return prediction
