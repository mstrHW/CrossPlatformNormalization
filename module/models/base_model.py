from abc import abstractmethod
import pickle
import types
import keras
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model, save_model
from sklearn.linear_model.base import BaseEstimator

from module.models.utils.regularizers import make_regularizer
from module.models.utils.optimizers import make_optimizer
from module.models.utils.metrics import make_metric, make_sklearn_metric
from module.models.utils.callbacks import TestHistoryCallback


class BaseModel(BaseEstimator):
    def __init__(self,
                 features_count,
                 layers=(100,),
                 activation='elu',
                 output_activation='linear',
                 drop_rate=0.,
                 regularizer_name='l1_l2',
                 regularizer_param=0.,
                 initializer='glorot_normal',
                 optimizer_name='adam',
                 epochs_count=200,
                 learning_rate=0.0001,
                 loss='mae',
                 batch_size=128,
                 patience=200,
                 learning_rate_decay_method='on_plato',
                 use_early_stopping=True,
                 ):

        self.features_count = features_count
        self.layers = layers
        self.activation = activation
        self.output_activation = output_activation
        self.drop_rate = drop_rate
        self.regularizer_name = regularizer_name
        self.regularizer_param = regularizer_param
        self.initializer = initializer
        self.optimizer_name = optimizer_name
        self.epochs_count = epochs_count
        self.learning_rate_decay_method = learning_rate_decay_method
        self.learning_rate = learning_rate
        self.loss = loss
        self.batch_size = batch_size
        self.patience = patience
        self.use_early_stopping = use_early_stopping

        self.regularizer = make_regularizer(regularizer_name, regularizer_param)
        self.model = self.build_model()

    def fit(self,
            train_generator,
            val_generator=None,
            test_generator=None,
            loss_history_file_name=None,
            model_checkpoint_file_name=None,
            tensorboard_log_dir=None,
            ):

        if test_generator is not None:
            callbacks_list = self.create_callbacks(
                model_checkpoint_file_name,
                tensorboard_log_dir,
                test_generator,
                steps_count=len(test_generator),
            )
        else:
            callbacks_list = self.create_callbacks(
                model_checkpoint_file_name,
                tensorboard_log_dir,
                val_generator,
                steps_count=len(val_generator),
            )

        history = self.model.fit_generator(
            train_generator,
            epochs=self.epochs_count,
            validation_data=val_generator,
            callbacks=callbacks_list,
            verbose=1,
            steps_per_epoch=len(train_generator),
            validation_steps=len(val_generator),
        )

        self.save_training_history(history, loss_history_file_name)

        return self

    def create_callbacks(self,
                         model_checkpoint_file_name,
                         tensorboard_log_dir,
                         test_data,
                         steps_count=0,
                         ):

        callbacks_list = []
        callbacks_list = self.add_early_stopping(callbacks_list, self.use_early_stopping)
        callbacks_list = self.add_reduce_on_plato(callbacks_list, self.use_early_stopping)
        callbacks_list = self.add_model_checkpoint(callbacks_list, model_checkpoint_file_name)

        if isinstance(test_data, keras.utils.Sequence):
            callbacks_list = self.add_test_score_generator(callbacks_list, tensorboard_log_dir, test_data, steps_count)
        else:
            callbacks_list = self.add_test_score(callbacks_list, tensorboard_log_dir, test_data)

        callbacks_list = self.add_tensorboard(callbacks_list, tensorboard_log_dir)

        return callbacks_list

    def add_early_stopping(self, callbacks_list, use_early_stopping):
        if use_early_stopping:
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                mode='min',
                verbose=1,
                patience=self.patience*2,
                restore_best_weights=True,
            )

            callbacks_list.append(early_stopping)

        return callbacks_list

    def add_reduce_on_plato(self, callbacks_list, use_early_stopping):
        if use_early_stopping:
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                patience=self.patience,
                # min_lr=self.learning_rate,
            )

            callbacks_list.append(reduce_lr)

        return callbacks_list

    def add_model_checkpoint(self, callbacks_list, model_checkpoint_file_name):
        if model_checkpoint_file_name is not None:
            checkpoint = ModelCheckpoint(
                filepath=model_checkpoint_file_name,
                save_best_only=True,
                monitor='val_loss')

            callbacks_list.append(checkpoint)

        return callbacks_list

    def add_tensorboard(self, callbacks_list, tensorboard_log_dir):
        if tensorboard_log_dir is not None:
            tensorboard_callback = keras.callbacks.TensorBoard(
                log_dir=tensorboard_log_dir,
                histogram_freq=0,
                write_graph=True,
                write_images=True,
            )
            callbacks_list.append(tensorboard_callback)

        return callbacks_list

    def add_test_score(self, callbacks_list, log_dir, test_data):
        if test_data is not None:
            score = TestHistoryCallback(
                log_dir=log_dir,
                test_data=test_data,
                scoring_method=lambda x: self.model.evaluate(*x)
            )
            callbacks_list.append(score)

        return callbacks_list

    def add_test_score_generator(self, callbacks_list, log_dir, test_data, steps_count):
        if test_data is not None:
            score = TestHistoryCallback(
                log_dir=log_dir,
                test_data=test_data,
                scoring_method=lambda x: self.model.evaluate_generator(
                    x,
                    steps=steps_count,
                )
            )

            callbacks_list.append(score)

        return callbacks_list

    def save_training_history(self, history, loss_history_file_name):
        if loss_history_file_name is not None:
            with open(loss_history_file_name, 'wb') as file:
                pickle.dump(history.history, file)

    def save_model(self, file_name):
        save_model(self.model, file_name)

    def load_model(self, file_name):
        self.model = load_model(
            file_name,
            custom_objects=dict(
                coeff_determination=make_metric('r2'),
            )
        )

    @abstractmethod
    def build_model(self):
        return None

    def predict(self, X):
        return self.model.predict(X)

    def predict_generator(self, X):
        y_preds = np.zeros(shape=(1, self.features_count))
        for batch in X:
            y_pred = self.model.predict(batch)
            y_preds = np.concatenate((y_preds, y_pred), axis=0)

        y_preds = y_preds[1:]

        return y_preds

    @abstractmethod
    def score(self, test_generator, metrics):
        if metrics is str:
            metrics = [metrics]

        y_preds = []
        ys = []
        for batch_X, batch_y in test_generator:
            batch_predicts = self.model.predict_on_batch(batch_X)
            y_preds.append(batch_predicts)

            ys.append(batch_y)

        if len(y_preds) > 1:
            y_preds = np.concatenate(y_preds, axis=0)
            ys = np.concatenate(ys, axis=0)
        else:
            y_preds = np.array(y_preds[0])
            ys = np.array(ys[0])

        scores = dict()
        for metric_name in metrics:
            scores[metric_name] = make_sklearn_metric(metric_name)(ys, y_preds)

        return scores

    def get_params(self, deep=True):
        return dict(
            features_count=self.features_count,
            layers=self.layers,
            activation=self.activation,
            output_activation=self.output_activation,
            drop_rate=self.drop_rate,
            regularizer_name=self.regularizer_name,
            regularizer_param=self.regularizer_param,
            initializer=self.initializer,
            optimizer_name=self.optimizer_name,
            epochs_count=self.epochs_count,
            learning_rate_decay_method=self.learning_rate_decay_method,
            learning_rate=self.learning_rate,
            loss=self.loss,
            batch_size=self.batch_size,
            patience=self.patience,
        )

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self

