from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import LearningRateScheduler
from module.models.base_model import BaseModel
import math


def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 100.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))

    return lrate


lrate = LearningRateScheduler(step_decay)


class DenoisingAutoencoder(BaseModel):
    def __init__(self, features_count, **kwargs):
        BaseModel.__init__(self, features_count, **kwargs)

    def build_model(self):
        input_layer = Input(shape=(self.features_count,))
        self.encoder = self.build_encoder(input_layer, self.layers[0], self.activation)
        self.decoder = self.build_decoder(self.encoder, self.layers[1], self.activation, self.output_activation)
        model = Model(input_layer, self.decoder)

        return model

    def build_encoder(self, input_layer, layers, activation):
        encoded = input_layer
        for layer in layers:
            encoded = Dense(layer, activation=activation)(encoded)

        return encoded

    def build_decoder(self, input_layer, layers, activation, output_activation):
        decoded = input_layer
        for layer in layers:
            decoded = Dense(layer, activation=activation)(decoded)

        decoded = Dense(self.features_count, activation=output_activation)(decoded)

        return decoded
