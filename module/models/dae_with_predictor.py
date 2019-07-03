from keras.layers import Input, Dense
from keras.models import Model
import numpy as np

from module.models.dae import DenoisingAutoencoder
from module.models.utils.metrics import make_sklearn_metric, make_metric
from module.models.utils.optimizers import make_optimizer
from module.data_processing.distance_noise_generation import DistanceNoiseGenerator
from module.data_processing.data_processing import get_batches


def get_train_generator(ref_data, corrupt_data, best_genes, noise_probability, batch_size, mode='train'):
    train_noised_generator = DistanceNoiseGenerator(
        ref_data,
        corrupt_data,
        best_genes,
        'train',
        noise_probability,
    )

    while True:     # For using with keras TODO: use keras generator class
        for batch in get_batches(ref_data, batch_size):
            corrupt_X = train_noised_generator.data_generation(batch[best_genes])
            y = batch[best_genes]
            age_column = batch['Age']
            yield corrupt_X, [y, age_column]
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
            age_column = batch['Age']
            yield X, [corrupt_y, age_column]
            if mode == 'test':
                return


class DAEwithPredictor(DenoisingAutoencoder):
    def __init__(self, features_count, **kwargs):
        DenoisingAutoencoder.__init__(self, features_count, **kwargs)
        self.predictor = None

    def build_model(self):
        input_layer = Input(shape=(self.features_count,))
        encoded = self.build_encoder(input_layer, self.layers[0], self.activation)
        decoded = self.build_decoder(encoded, self.layers[1], self.activation, self.output_activation)
        predicted = Dense(1, activation=self.output_activation, name='predictor')(encoded)

        model = Model(input_layer, [decoded, predicted])

        opt = make_optimizer(self.optimizer_name, lr=self.learning_rate)
        model.compile(loss=self.loss, optimizer=opt, metrics=[make_metric('r2')])

        return model

    def fit(self,
            train_data,
            best_genes,
            val_data=None,
            test_data=None,
            loss_history_file_name=None,
            model_checkpoint_file_name=None,
            tensorboard_log_dir=None,
            ):

        ref_batch_name = train_data['GEO'].value_counts().keys()[0]
        ref_mask = train_data['GEO'] == ref_batch_name

        ref_data = train_data[ref_mask]
        corrupt_data = train_data[~ref_mask]

        train_generator = get_train_generator(
            ref_data,
            corrupt_data,
            best_genes,
            self.noise_probability,
            self.batch_size,
        )

        val_generator = get_test_generator(
            ref_data,
            val_data,
            best_genes,
            self.noise_probability,
            self.batch_size,
        )

        if test_data is not None:
            test_generator = get_test_generator(
                ref_data,
                test_data,
                best_genes,
                self.noise_probability,
                self.batch_size,
            )

            callbacks_list = self.create_callbacks(
                model_checkpoint_file_name,
                tensorboard_log_dir,
                test_generator,
                steps_count=test_data.shape[0] / self.batch_size,
            )
        else:
            callbacks_list = self.create_callbacks(
                model_checkpoint_file_name,
                tensorboard_log_dir,
                val_generator,
                steps_count=val_data.shape[0] / self.batch_size,
            )

        history = self.model.fit_generator(
            train_generator,
            epochs=self.epochs_count,
            validation_data=val_generator,
            callbacks=callbacks_list,
            verbose=1,
            steps_per_epoch=ref_data.shape[0] / self.batch_size,
            validation_steps=val_data.shape[0] / self.batch_size,
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
