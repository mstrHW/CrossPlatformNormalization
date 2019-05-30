import logging

from module.models.mlp import MLP
from module.data_processing import data_processing
from module.models.grid_search import search_parameters
from module.models.utils import save_results, save_search_results

logging.basicConfig(level=logging.DEBUG)


def main():
    logging.debug('Read data')
    data_params = dict(
        features_count=1000,
        tissues=['Whole blood'],
        normalize=True,
        noising_method=None,
        batch_size=512,
        rows_count=None
    )

    train_data, test_data = data_processing.main(False, **data_params)

    logging.debug('Create model')

    model_params = dict(
        layers=(1500, 1200, 1000, 800, 700, 500, 256, 128),
        activation='elu',
        drop_rate=0.5,
        regularizer_method='l1_l2',
        regularizers_param=1e-3,
        epochs_count=200,
        learning_rate=0.0001,
        loss='mae',
        batch_size=data_params['batch_size'],
        patience=200,
    )

    model = MLP(features_count=data_params['features_count'], **model_params)


    learning_params = dict(
        use_early_stopping=True,
        loss_history_file_name=None,
        model_checkpoint_file_name=None,
    )

    logging.debug('Fit model')
    model.fit(train_data, test_data)

    train_score = model.calculate_score(train_data[0], train_data[1])
    test_score = model.calculate_score(test_data[0], test_data[1])

    save_results('MLP', data_params, model_params, train_score, test_score)


def search_model_parameters():
    import random
    from datetime import datetime

    now = datetime.strptime('2019-05-29 14:38:08.916814', '%Y-%m-%d %H:%M:%S.%f')
    print('now : ', now)
    random.seed(now)
    np_seed = 5531
    sklearn_seed = 23

    import numpy as np
    import tensorflow as tf

    np.random.seed(np_seed)
    tf.random.set_random_seed(np_seed)

    logging.debug('Read data')
    data_params = dict(
        features_count=1000,
        tissues=['Whole blood'],
        normalize=True,
        noising_method=None,
        batch_size=512,
        rows_count=None,
    )

    train_data, test_data = data_processing.main(False, **data_params)

    logging.debug('Start grid search')

    layers = (1000, 800, 600, 300, 128, 10)
    activation = ['elu', 'relu', 'tanh']  # softmax, softplus, softsign, 'sigmoid', 'hard_sigmoid', 'linear'
    dropout_rate = [0.0, 0.7]   #, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
    epochs_count = 20,
    learning_rate = 0.0001,
    loss = 'mae',

    model_parameters_space = dict(
        drop_rate=dropout_rate,
        activation=activation,
        epochs_count=epochs_count,
        learning_rate=learning_rate,
        loss=loss,
    )

    learning_params = dict(
        use_early_stopping=True,
        # loss_history_file_name=,
        # model_checkpoint_file_name=None,
    )

    gs_results = search_parameters(
        MLP(features_count=1000, layers=layers),
        (train_data, test_data),
        model_parameters_space,
        learning_params,
        dict(cv=3, n_jobs=1, return_train_score=True, random_state=sklearn_seed),
    )

    save_search_results(*gs_results, 'MLP_GS')


if __name__ == '__main__':
    # main()
    search_model_parameters()
