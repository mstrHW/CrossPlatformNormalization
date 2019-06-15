import logging

from module.models.mlp import MLP
from module.data_processing.data_processing import load_data, filter_data, get_train_test, get_X_y
from sklearn.preprocessing import MinMaxScaler
from module.models.grid_search import search_parameters
from definitions import *
import json
from imp import reload
reload(logging)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    logging.basicConfig(level=logging.DEBUG, filename=r'log.log')
    logging.debug('Read data')
    data_params = dict(
        features_count=1000,
        rows_count=None,
        filtered_column='Tissue',
        using_values='Whole blood',
        target_column='Age',
        normalize=True,
        use_generator=False,
        noising_method=None,
        batch_size=128,
    )

    input_data, best_genes = load_data(data_params['features_count'])
    input_data = filter_data(input_data, data_params['filtered_column'], data_params['using_values'])

    train_data, test_data = get_train_test(input_data)
    train_X, train_y = get_X_y(train_data, using_genes=best_genes, target_column=data_params['target_column'])
    test_X, test_y = get_X_y(train_data, using_genes=best_genes, target_column=data_params['target_column'])

    scaler = None
    if data_params['normalize']:
        scaler = MinMaxScaler()
        scaler.fit(input_data[best_genes])

        train_X = scaler.transform(train_X)
        test_X = scaler.transform(test_X)

    logging.debug('Create model')

    model_params = dict(
        layers=(1500, 800, 700, 500, 128, 1),
        activation='elu',
        drop_rate=0.5,
        regularizer_name='l1_l2',
        regularizer_param=1e-3,
        epochs_count=100,
        learning_rate=0.00001,
        loss='mae',
        batch_size=data_params['batch_size'],
        patience=200,
        optimizer_name='eve',
    )

    model = MLP(features_count=data_params['features_count'], **model_params)

    learning_params = dict(
        use_early_stopping=True,
        loss_history_file_name=None,
        model_checkpoint_file_name=None,
    )

    logging.debug('Fit model')
    model.fit(
        *(train_X, train_y),
        (test_X, test_y),
        **learning_params,
    )


def search_model_parameters():
    experiment_dir = os.path.join(MODELS_DIR, 'predict_age/mlp')
    make_dirs(experiment_dir)

    log_dir = os.path.join(experiment_dir, 'log')
    make_dirs(log_dir)

    log_name = 'log.log'
    logging.basicConfig(level=logging.DEBUG, filename=os.path.join(log_dir, log_name))
    logging.info('Read data')

    data_params = dict(
        features_count=1000,
        rows_count=None,
        filtered_column='Tissue',
        using_values='Whole blood',
        target_column='Age',
        normalize=True,
        use_generator=False,
        noising_method=None,
        batch_size=128,
    )

    experiment_meta_params_file = os.path.join(experiment_dir, 'experiment_meta_parameters.json')
    with open(experiment_meta_params_file, 'w') as file:
        write_message = dict(
            np_seed=np_seed,
            sklearn_seed=sklearn_seed,
            data_params=data_params,
        )
        json.dump(write_message, file)
        logging.info('experiment meta parameters was saved at file {}'.format(experiment_meta_params_file))

    input_data, best_genes = load_data(data_params['features_count'])
    input_data = filter_data(input_data, data_params['filtered_column'], data_params['using_values'])

    # train_X, train_y = get_X_y(train_data, using_genes=best_genes, target_column=data_params['target_column'])
    # test_X, test_y = get_X_y(train_data, using_genes=best_genes, target_column=data_params['target_column'])
    #
    # scaler = None
    # if data_params['normalize']:
    #     scaler = MinMaxScaler()
    #     scaler.fit(input_data[best_genes])
    #
    #     train_X = scaler.transform(train_X)
    #     test_X = scaler.transform(test_X)

    logging.info('Start grid search')

    layers = [
        (256, 128, 64, 1),
        (512, 256, 128, 1),
        (512, 384, 256, 128, 1),
        (768, 512, 384, 192, 1),
        (1024, 768, 512, 384, 128, 1),
        (1536, 1024, 768, 384, 192, 1),
    ]
    activation = ['elu', 'lrelu', 'prelu']
    dropout_rate = [0.25, 0.5, 0.75]
    regularization_param = [10 ** -i for i in range(3, 7)]
    epochs_count = 2000,
    loss = 'mae',
    optimizer = ['adam', 'rmsprop'] #, 'eve's

    model_parameters_space = dict(
        layers=layers,
        drop_rate=dropout_rate,
        activation=activation,
        regularizer_param=regularization_param,
        epochs_count=epochs_count,
        loss=loss,
        optimizer_name=optimizer,
    )

    learning_params = dict(
        use_early_stopping=True,
    )

    search_parameters(
        lambda **kwargs: MLP(features_count=data_params['features_count'], **kwargs),
        input_data,
        best_genes,
        data_params,
        using_metrics=['r2', 'mae'],
        model_parameters_space=model_parameters_space,
        learning_parameters=learning_params,
        cross_validation_parameters=dict(n_splits=5, random_state=sklearn_seed, shuffle=True),
        experiment_dir=experiment_dir,
        results_file='cv_results.json',
        search_method_name='random',
        random_n_iter=200,
    )


if __name__ == '__main__':
    # main()
    search_model_parameters()
