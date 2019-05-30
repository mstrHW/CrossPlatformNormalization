from abc import ABC, abstractmethod
import pickle
import inspect
import keras
from keras.callbacks import ModelCheckpoint
from keras.models import load_model, save_model
from sklearn.linear_model.base import BaseEstimator
from keras.wrappers.scikit_learn import KerasRegressor
from module.models.utils import make_regularizer
from module.models.optimizers import make_optimizer


class BaseModel(BaseEstimator):
    def __init__(self,
                 features_count,
                 layers=(100,),
                 activation='elu',
                 output_activation='linear',
                 drop_rate=0.,
                 regularizer_name='l1_l2',
                 regularizer_param=0.,
                 initializer='glorot_uniform',
                 optimizer_name='adam',
                 epochs_count=200,
                 learning_rate=0.0001,
                 loss='mae',
                 batch_size=128,
                 patience=200,
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
        self.learning_rate = learning_rate
        self.loss = loss
        self.batch_size = batch_size
        self.patience = patience

        self.regularizer = make_regularizer(regularizer_name, regularizer_param)
        self.model = self.build_model()

        opt = make_optimizer(self.optimizer_name, lr=self.learning_rate)
        self.model.compile(loss=self.loss, optimizer=opt)

    def fit(self,
            train_X,
            train_y,
            test_data=None,
            use_early_stopping=False,
            loss_history_file_name=None,
            model_checkpoint_file_name=None,
            ):

        callbacks_list = []
        callbacks_list = self.add_early_stopping(callbacks_list, use_early_stopping)
        callbacks_list = self.add_model_checkpoint(callbacks_list, model_checkpoint_file_name)

        history = self.model.fit(
            train_X,
            train_y,
            epochs=self.epochs_count,
            batch_size=self.batch_size,
            validation_data=test_data,
            callbacks=callbacks_list,
        )

        self.save_training_history(history, loss_history_file_name)

        return self

    def add_early_stopping(self, callbacks_list, use_early_stopping):
        if use_early_stopping:
            early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                           mode='min',
                                                           verbose=10,
                                                           patience=self.patience,
                                                           restore_best_weights=True)

            callbacks_list.append(early_stopping)

        return callbacks_list

    def add_model_checkpoint(self, callbacks_list, model_checkpoint_file_name):
        if model_checkpoint_file_name is not None:
            checkpoint = ModelCheckpoint(
                filepath=model_checkpoint_file_name,
                save_best_only=True,
                monitor='val_loss')

            callbacks_list.append(checkpoint)

        return callbacks_list

    def save_training_history(self, history, loss_history_file_name):
        if loss_history_file_name is not None:
            with open(loss_history_file_name, 'wb') as file:
                pickle.dump(history.history, file)

    def fit_generator(self,
                      training_generator,
                      testing_generator,
                      use_early_stopping=False,
                      loss_history_file_name=None,
                      # model_checkpoint_file_name=None,
                      ):

        callbacks_list = []
        callbacks_list = self.add_early_stopping(callbacks_list, use_early_stopping)
        # callbacks_list = self.add_model_checkpoint(callbacks_list, model_checkpoint_file_name)

        history = self.model.fit_generator(
            generator=training_generator,
            validation_data=testing_generator,
            epochs=self.epochs_count,
            callbacks=callbacks_list,
        )

        self.save_training_history(history, loss_history_file_name)

    def save_model(self, file_name):
        save_model(self.model, file_name)

    def load_model(self, file_name):
        self.model = load_model(file_name)

    @abstractmethod
    def build_model(self):
        return None

    def predict(self, X):
        return self.model.predict(X)

    def predict_generator(self, generator):
        return self.model.predict_generator(generator)

    def score(self, X, y):
        return self.model.evaluate(X, y)

    def score_generator(self, generator):
        return self.model.evaluate_generator(generator)

    def get_params(self, deep=True):
        return dict(
            features_count=self.features_count,
            layers=self.layers,
            activation=self.activation,
            output_activation=self.output_activation,
            drop_rate=self.drop_rate,
            regularizers_param=self.regularizers_param,
            initializer=self.initializer,
            optimizer_name=self.optimizer_name,
            epochs_count=self.epochs_count,
            learning_rate=self.learning_rate,
            loss=self.loss,
        )

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self

