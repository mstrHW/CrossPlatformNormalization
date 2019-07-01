from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.models import Model
import math

from module.models.base_model import BaseModel
from module.models.utils.activations import make_activation
from module.models.utils.optimizers import make_optimizer
from module.models.utils.metrics import make_metric
from module.data_processing.data_generator import DistanceNoiseGenerator, get_batches


def get_train_generator(ref_data, corrupt_data, best_genes, noise_probability, batch_size):
    train_noised_generator = DistanceNoiseGenerator(
        ref_data,
        corrupt_data,
        best_genes,
        'train',
        noise_probability,
    )

    for batch in get_batches(ref_data, batch_size):
        corrupt_X = train_noised_generator.data_generation(batch)
        y = batch
        yield corrupt_X, y


def get_test_generator(ref_data, corrupt_data, best_genes, noise_probability, batch_size):
    train_noised_generator = DistanceNoiseGenerator(
        ref_data,
        corrupt_data,
        best_genes,
        'test',
        noise_probability,
    )

    for batch in get_batches(ref_data, batch_size):
        X = batch
        corrupt_y = train_noised_generator.data_generation(batch)
        yield X, corrupt_y


class DenoisingAutoencoder(BaseModel):
    def __init__(self, features_count, noise_probability, **kwargs):
        BaseModel.__init__(self, features_count, **kwargs)
        self.encoder = None
        self.decoder = None
        self.noise_probability = noise_probability

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

    def fit(self,
            train_data,
            val_data=None,
            test_data=None,
            loss_history_file_name=None,
            model_checkpoint_file_name=None,
            tensorboard_log_dir=None,
            ):

        train_X, train_y = train_data
        val_X, val_y = val_data
        test_X, test_y = test_data

        ref_batch_name = train_X['GEO'].value_counts().keys()[0]
        ref_mask = train_X['GEO'] == ref_batch_name

        columns = train_X.columns
        columns = columns.drop(['GEO'])
        best_genes = columns.tolist()

        ref_data = train_X[ref_mask]
        corrupt_data = train_X[~ref_mask]

        train_generator = get_train_generator(
            ref_data,
            corrupt_data,
            best_genes,
            self.noise_probability,
            self.batch_size,
        )

        val_generator = get_test_generator(
            ref_data,
            val_X,
            best_genes,
            self.noise_probability,
            self.batch_size,
        )

        test_generator = get_test_generator(
            ref_data,
            test_X,
            best_genes,
            self.noise_probability,
            self.batch_size,
        )

        callbacks_list = self.create_callbacks(
            model_checkpoint_file_name,
            tensorboard_log_dir,
            test_generator,
        )

        history = self.model.fit_generator(
            train_generator,
            epochs=self.epochs_count,
            validation_data=val_generator,
            callbacks=callbacks_list,
            verbose=1,
            steps_per_epoch=ref_data.shape[0] / self.batch_size,
            validation_steps=val_X.shape[0] / self.batch_size,
        )

        self.save_training_history(history, loss_history_file_name)

        return self
