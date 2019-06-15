from keras.layers import Input, Dense
from keras.models import Model
from module.models.dae import DenoisingAutoencoder


class DAEwithPredictor(DenoisingAutoencoder):
    def __init__(self, features_count, **kwargs):
        DenoisingAutoencoder.__init__(self, features_count, **kwargs)
        self.predictor = None

    def build_model(self):
        input_layer = Input(shape=(self.features_count,))
        self.encoder = self.build_encoder(input_layer, self.layers[0], self.activation)
        self.decoder = self.build_decoder(self.encoder, self.layers[1], self.activation, self.output_activation)
        self.predictor = Dense(1, activation=self.output_activation)(self.encoder)
        model = Model(input_layer, [self.decoder, self.predictor])

        return model
