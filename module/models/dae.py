from keras.layers import Input, Dense
from keras.models import Model

from module.models.base_model import BaseModel
from module.models.utils.activations import make_activation


class DenoisingAutoencoder(BaseModel):
    def __init__(self, features_count, **kwargs):
        BaseModel.__init__(self, features_count, **kwargs)
        self.encoder = None
        self.decoder = None

    def build_model(self):
        input_layer = Input(shape=(self.features_count,))
        self.encoder = self.build_encoder(input_layer, self.layers[0], self.activation)
        self.decoder = self.build_decoder(self.encoder, self.layers[1], self.activation, self.output_activation)
        model = Model(input_layer, self.decoder)

        return model

    def build_encoder(self, input_layer, layers, activation):
        encoded = input_layer
        # encoded = BatchNormalization()(encoded)
        for layer in layers:
            encoded = Dense(layer)(encoded)
            encoded = make_activation(activation)(encoded)
            # encoded = BatchNormalization()(encoded)

        return encoded

    def build_decoder(self, input_layer, layers, activation, output_activation):
        decoded = input_layer
        for layer in layers:
            decoded = Dense(layer)(decoded)
            decoded = make_activation(activation)(decoded)
            # decoded = BatchNormalization()(decoded)

        decoded = Dense(self.features_count, activation=output_activation)(decoded)

        return decoded
