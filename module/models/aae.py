from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model
import numpy as np

from module.models.dae import DenoisingAutoencoder
from module.models.gan import GenerativeAdversarialNetwork
from module.models.utils.metrics import make_sklearn_metric
from module.models.utils.activations import make_activation
from module.models.utils.optimizers import make_optimizer


class AdversarialAutoencoder(DenoisingAutoencoder, GenerativeAdversarialNetwork):
    def __init__(self, features_count, latent_dim, **kwargs):
        DenoisingAutoencoder.__init__(self, features_count, **kwargs)
        self.latent_dim = latent_dim

    def build_model(self):
        self.encoder = self.build_encoder(self.features_count, self.layers[0], self.activation)
        self.decoder = self.build_decoder(self.latent_dim, self.layers[1], self.activation, self.output_activation)

        self.discriminator = self.build_discriminator(self.latent_dim, self.layers[2], self.activation)

        input_layer = Input(shape=self.features_count)
        encoded = self.encoder(input_layer)
        decoded = self.decoder(encoded)

        self.discriminator.trainable = False
        validity = self.discriminator(encoded)

        model = Model(input_layer, [decoded, validity])
        model.compile(
            loss=['mae', 'binary_crossentropy'],
            loss_weights=[0.999, 0.001],
            optimizer=make_optimizer(self.optimizer_name),
        )

        return model

    def fit(self,
            train_data,
            val_data=None,
            test_data=None,
            use_early_stopping=False,
            loss_history_file_name=None,
            model_checkpoint_file_name=None,
            tensorboard_log_dir=None,
            ):

        train_X, train_y = train_data
        val_X, val_y = val_data

        for epoch in range(self.epochs_count):
            noise = np.random.normal(0, 1, [self.batch_size, self.latent_dim])
            generated_examples = self.encoder.predict(noise)

            examples_batch = train_X[np.random.randint(low=0, high=train_X.shape[0], size=self.batch_size)]

            X = np.concatenate([examples_batch, generated_examples])

            y_dis = np.zeros(2 * self.batch_size)
            y_dis[:self.batch_size] = 0.9

            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)

            noise = np.random.normal(0, 1, [self.batch_size, latent_dim])
            y_gen = np.ones(self.batch_size)

            discriminator.trainable = False
            gan.train_on_batch(noise, y_gen)

    def score(self, test_data, metrics, scaler=None):
        if metrics is str:
            metrics = [metrics]

        y_preds = self.model.predict(test_data[0])
        decoder_pred, predictor_pred = y_preds

        ys = test_data[1]
        decoder_y, predictor_y = ys

        if scaler is not None:
            decoder_pred = scaler.inverse_transform(decoder_pred)
            decoder_y = scaler.inverse_transform(decoder_y)

        scores = dict()
        for metric_name in metrics:
            scores['decoder_{}'.format(metric_name)] = make_sklearn_metric(metric_name)(decoder_y, decoder_pred)

        for metric_name in metrics:
            scores['predictor_{}'.format(metric_name)] = make_sklearn_metric(metric_name)(predictor_y, predictor_pred)

        return scores


if __name__ == '__main__':
    aae = AdversarialAutoencoder()
    aae.train(epochs=20000, batch_size=32, sample_interval=200)
