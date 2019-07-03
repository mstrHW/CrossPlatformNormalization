from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.models import Model
import math
import numpy as np

from module.models.base_model import BaseModel
from module.models.utils.activations import make_activation
from module.models.utils.optimizers import make_optimizer
from module.models.utils.metrics import make_metric, make_sklearn_metric
from module.data_processing.NoisedDataGeneration import DistanceNoiseGenerator
from module.data_processing.data_processing import get_batches


def get_train_generator(ref_data, corrupt_data, best_genes, noise_probability, batch_size, mode='train'):
    train_noised_generator = DistanceNoiseGenerator(
        ref_data,
        corrupt_data,
        best_genes,
        'train',
        noise_probability,
    )

    while True:
        for batch in get_batches(ref_data, batch_size):
            corrupt_X = train_noised_generator.data_generation(batch[best_genes])
            y = batch[best_genes]
            yield corrupt_X, y
            if mode == 'test':
                return


def get_test_generator(ref_data, corrupt_data, best_genes, noise_probability, batch_size, mode='train'):
    train_noised_generator = DistanceNoiseGenerator(
        ref_data,
        corrupt_data,
        best_genes,
        'test',
        noise_probability,
    )

    while True:
        for batch in get_batches(ref_data, batch_size):
            X = batch[best_genes]
            corrupt_y = train_noised_generator.data_generation(batch[best_genes])
            yield X, corrupt_y
            if mode == 'test':
                return


class DenoisingAutoencoder(BaseModel):
    def __init__(self, features_count, noise_probability=0.25, **kwargs):
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
            steps_count=test_X.shape[0] / self.batch_size,
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

        self.ref_data = ref_data
        self.best_genes = best_genes
        self.save_training_history(history, loss_history_file_name)

        return self

    def score(self, X, y, metrics, mode='train'):
        if metrics is str:
            metrics = [metrics]

        if mode == 'train':
            ref_batch_name = X['GEO'].value_counts().keys()[0]
            ref_mask = X['GEO'] == ref_batch_name

            corrupt_data = X[~ref_mask]

            generator = get_train_generator(
                self.ref_data,
                corrupt_data,
                self.best_genes,
                self.noise_probability,
                self.batch_size,
                'test',
            )
        else:
            generator = get_test_generator(
                self.ref_data,
                X,
                self.best_genes,
                self.noise_probability,
                self.batch_size,
                'test',
            )

        y_preds = []
        ys = []
        for batch_X, batch_y in generator:
            batch_predicts = self.model.predict_on_batch(batch_X)
            y_preds.append(batch_predicts)

            ys.append(batch_y)

        y_preds = np.concatenate(y_preds, axis=0)
        ys = np.concatenate(ys, axis=0)

        scores = dict()
        for metric_name in metrics:
            scores[metric_name] = make_sklearn_metric(metric_name)(ys, y_preds)

        return scores

    def get_params(self, deep=True):
        params = BaseModel.get_params(self, deep)
        params['noise_probability'] = self.noise_probability

        return params