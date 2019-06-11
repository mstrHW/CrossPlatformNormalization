from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model
from keras.callbacks import LearningRateScheduler
from module.models.base_model import BaseModel, make_activation
import math
from module.models.optimizers import make_optimizer
from sklearn.metrics import r2_score


def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 100.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))

    return lrate


def coeff_determination(y_true, y_pred):
    from keras import backend as K
    SS_res = K.sum(K.square(y_true - y_pred), axis=0)
    SS_tot = K.sum(K.square(y_true - K.mean(y_true, axis=0)), axis=0)
    r2_vect = (1 - SS_res/(SS_tot + K.epsilon()))
    result_mean = K.mean(r2_vect)
    return result_mean


lrate = LearningRateScheduler(step_decay)


class DenoisingAutoencoder(BaseModel):
    def __init__(self, features_count, **kwargs):
        BaseModel.__init__(self, features_count, **kwargs)

    def build_model(self):
        input_layer = Input(shape=(self.features_count,))
        self.encoder = self.build_encoder(input_layer, self.layers[0], self.activation)
        self.decoder = self.build_decoder(self.encoder, self.layers[1], self.activation, self.output_activation)
        model = Model(input_layer, self.decoder)

        opt = make_optimizer(self.optimizer_name, lr=self.learning_rate)
        model.compile(loss=self.loss, optimizer=opt, metrics=['mae', coeff_determination])

        return model

    def build_encoder(self, input_layer, layers, activation):
        encoded = input_layer
        # encoded = BatchNormalization()(encoded)
        for layer in layers:
            encoded = Dense(layer)(encoded)
            encoded = make_activation(self.activation)(encoded)
            # encoded = BatchNormalization()(encoded)

        return encoded

    def build_decoder(self, input_layer, layers, activation, output_activation):
        decoded = input_layer
        for layer in layers:
            decoded = Dense(layer)(decoded)
            decoded = make_activation(self.activation)(decoded)
            # decoded = BatchNormalization()(decoded)

        decoded = Dense(self.features_count, activation=output_activation)(decoded)

        return decoded
