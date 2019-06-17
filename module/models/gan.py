from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model
import numpy as np

from module.models.base_model import BaseModel
from module.models.utils.metrics import make_sklearn_metric
from module.models.utils.activations import make_activation
from module.models.utils.optimizers import make_optimizer


class GenerativeAdversarialNetwork(BaseModel):
    def __init__(self, features_count, **kwargs):
        BaseModel.__init__(self, features_count, **kwargs)
        self.discriminator = None

    def build_generator(self, latent_dim, output_dim, layers, activation, output_actvation):
        input_layer = Input(shape=(latent_dim,))
        generator = input_layer
        for layer in layers:
            generator = Dense(layer)(generator)
            generator = make_activation(activation)(generator)
            generator = BatchNormalization()(generator)

        generator.add(Dense(units=output_dim, activation=output_actvation))
        generator.compile(
            loss='binary_crossentropy',
            optimizer=make_optimizer(self.optimizer_name),
        )

        return generator

    def build_discriminator(self, latent_dim, layers, activation):
        input_layer = Input(shape=(latent_dim,))
        discriminator = input_layer
        for layer in layers:
            discriminator = Dense(layer)(discriminator)
            discriminator = make_activation(activation)(discriminator)
            discriminator = BatchNormalization()(discriminator)

        discriminator = Dense(1, activation='sigmoid', name='discriminator')(discriminator)
        discriminator = Model(input_layer, discriminator)
        discriminator.compile(
            loss='binary_crossentropy',
            optimizer=make_optimizer(self.optimizer_name),
        )

        return discriminator

    def build_model(self, latent_dim, discriminator, generator):
        discriminator.trainable = False

        gan_input = Input(shape=(latent_dim,))
        generated = generator(gan_input)
        gan_output = discriminator(generated)

        gan = Model(inputs=gan_input, outputs=gan_output)
        gan.compile(
            loss='binary_crossentropy',
            optimizer=make_optimizer(self.optimizer_name),
        )

        return gan

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

        latent_dim = 100
        generator = self.build_generator(latent_dim, train_X.shape[1], self.layers[0], self.activation, self.output_activation)
        discriminator = self.build_discriminator(latent_dim, self.layers[1], self.activation)
        gan = self.build_model(latent_dim, discriminator, generator)

        for i in range(self.epochs_count):

                noise = np.random.normal(0, 1, [self.batch_size, latent_dim])
                generated_examples = generator.predict(noise)

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
