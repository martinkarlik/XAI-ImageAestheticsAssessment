import importlib

from keras.layers import Dropout, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay


class BaseModuleBP:

    def __init__(self, custom_weights=None, weights='imagenet', base_model_name="InceptionResNetV2", n_classes=1,
                 learning_rate=0.001, dropout_rate=0, decay_rate=0.9, decay_steps=10000, loss="mean_squared_error"):

        self.base_model_name = base_model_name
        self.n_classes = n_classes
        self.weights = weights
        self.custom_weights = custom_weights

        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.loss = loss

        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

        if self.base_model_name == "InceptionResNetV2":
            self.base_module = importlib.import_module('tensorflow.keras.applications.inception_resnet_v2')
        elif self.base_model_name == "InceptionV3":
            self.base_module = importlib.import_module('tensorflow.keras.applications.inception_v3')
        else:
            self.base_module = importlib.import_module('tensorflow.keras.applications.' + self.base_model_name.lower())

        self.base_model = None
        self.bp_model = None

    def build(self):
        imagenet_cnn = getattr(self.base_module, self.base_model_name)

        # Replace last layer with Dropout and Dense (virtually turn classification into multi-output regression).
        self.base_model = imagenet_cnn(input_shape=(224, 224, 3), weights=self.weights, include_top=False, pooling='avg')

        x = Dropout(self.dropout_rate)(self.base_model.output)
        x = Dense(units=self.n_classes, activation='sigmoid')(x)

        self.bp_model = Model(self.base_model.inputs, x)
        if self.custom_weights:
            self.bp_model.load_weights(self.custom_weights)

    def compile(self):
        lr_schedule = ExponentialDecay(
            self.learning_rate, decay_steps=self.decay_steps, decay_rate=self.decay_rate, staircase=True
        )

        self.bp_model.compile(
            optimizer=Adam(learning_rate=lr_schedule), loss=self.loss)

    def predict(self, image):
        self.bp_model.predict(image)

    def preprocess(self):
        return self.base_module.preprocess_input