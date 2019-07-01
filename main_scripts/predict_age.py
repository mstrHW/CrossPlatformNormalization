import logging

from module.models.mlp import MLP
from module.models.utils.grid_search import search_parameters, choose_cross_validation
from module.data_processing.data_generating_cases import processing_conveyor
from module.data_processing.data_processing import get_train_test
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
    experiment_dir = os.path.join(MODELS_DIR, 'predict_age/mlp/test/trained')
    make_dirs(experiment_dir)

    log_dir = os.path.join(experiment_dir, 'log')
    make_dirs(log_dir)

    log_name = 'log.log'
    logging.basicConfig(level=logging.DEBUG, filename=os.path.join(log_dir, log_name))
    logging.info('Read data')

    processing_sequence = {
        'load_test_data': dict(
            features_count=1000,
            rows_count=None,
        ),
        'filter_data': dict(
            filtered_column='Tissue',
            using_values='Whole blood',
        ),
        'normalization': dict(
            method='series',
        ),
    }

    data_wrapper = processing_conveyor(processing_sequence)
    train_data, test_data = get_train_test(data_wrapper.processed_data)

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

    cross_validation_method_name = 'custom'
    cross_validation_parameters = dict(
        n_splits=5,
        random_state=sklearn_seed,
        shuffle=True,
    )

    experiment_meta_params_file = os.path.join(experiment_dir, 'experiment_meta_parameters.json')
    with open(experiment_meta_params_file, 'w') as file:
        write_message = dict(
            np_seed=np_seed,
            sklearn_seed=sklearn_seed,
        )
        json.dump(write_message, file)
        logging.info('experiment meta parameters was saved at file {}'.format(experiment_meta_params_file))

    data_parameters_file = os.path.join(experiment_dir, 'data_parameters.json')
    with open(data_parameters_file, 'w') as file:
        write_message = dict(
            data_processing=processing_sequence,
            cross_validation_method=cross_validation_method_name,
            cross_validation=cross_validation_parameters,
        )
        json.dump(write_message, file)
        logging.info('experiment meta parameters was saved at file {}'.format(data_parameters_file))

    cross_validation_method = choose_cross_validation(cross_validation_method_name)
    get_x_y_method = lambda x: (x[data_wrapper.best_genes], x['Age'])

    search_parameters(
        lambda **params: MLP(data_wrapper.processing_sequence['load_test_data']['features_count'], **params),
        train_data,
        test_data,
        cross_validation_method,
        cross_validation_parameters,
        get_x_y_method,
        ['r2'],
        model_parameters_space,
        experiment_dir,
        'cv_results_mlp_test.json',
        'random',
    )


if __name__ == '__main__':
    # main_scripts()
    search_model_parameters()
