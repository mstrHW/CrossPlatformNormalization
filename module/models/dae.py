from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.models import Model

from module.models.base_model import BaseModel
from module.models.utils.activations import make_activation
from module.models.utils.optimizers import make_optimizer
from module.models.utils.metrics import make_metric


class DenoisingAutoencoder(BaseModel):
    def __init__(self, features_count, **kwargs):
        BaseModel.__init__(self, features_count, **kwargs)
        self.encoder = None
        self.decoder = None

    def build_model(self):
        input_layer = Input(shape=(self.features_count,))
        encoded = self.build_encoder(input_layer, self.layers[0], self.activation)
        decoded = self.build_decoder(encoded, self.layers[1], self.activation, self.output_activation)

        model = Model(input_layer, decoded)
        opt = make_optimizer(self.optimizer_name, lr=self.learning_rate)
        model.compile(loss=self.loss, optimizer=opt, metrics=[make_metric('r2')])

        return model

    def build_encoder(self, input_layer, layers, activation):
        encoded = input_layer
        encoded = BatchNormalization()(encoded)
        for layer in layers:
            encoded = Dense(layer)(encoded)
            encoded = make_activation(activation)(encoded)
            encoded = BatchNormalization()(encoded)

        return encoded

    def build_decoder(self, input_layer, layers, activation, output_activation):
        decoded = input_layer
        for layer in layers:
            decoded = Dense(layer)(decoded)
            decoded = make_activation(activation)(decoded)
            decoded = BatchNormalization()(decoded)

        decoded = Dense(self.features_count, activation=output_activation, name='decoder')(decoded)

        return decoded
