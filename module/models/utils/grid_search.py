import os
import numpy as np
import json
import pandas as pd
import logging

from keras import backend as K
import tensorflow as tf
from sklearn.model_selection import ParameterGrid, ParameterSampler

import definitions
from module.data_processing.data_processing import get_train_test


def search_parameters(model_class,
                      train_data,
                      test_data,
                      cross_validation_method,
                      cross_validation_parameters,
                      get_generator,
                      using_metrics,
                      model_parameters_space,
                      experiment_dir,
                      results_file,
                      search_method_name='grid',
                      random_n_iter=150,
                      ):

    logging.info('Start search method : {}'.format(search_method_name))

    choose_search_method = {
        'grid': ParameterGrid,
        'random': lambda kwargs: ParameterSampler(kwargs, n_iter=random_n_iter, random_state=definitions.sklearn_seed)
    }

    search_method = choose_search_method[search_method_name]
    parameters_generator = search_method(model_parameters_space)

    search_results_file = os.path.join(experiment_dir, results_file)
    parameters_generator, search_results = __restore_search_state(parameters_generator, search_results_file)

    cv_splits_count = cross_validation_parameters['n_splits']

    trained_models_dir = os.path.join(experiment_dir, 'trained_models')
    definitions.make_dirs(trained_models_dir)
    logging.info('trained models dir created at path : {}'.format(trained_models_dir))

    for params in parameters_generator:
        print(params)
        logging.info('current parameters : {}'.format(params))

        model_dir = __create_model_directory(trained_models_dir)
        logging.info('model directory : {}'.format(model_dir))

        cv_scores = np.zeros((cv_splits_count, 3))
        model = model_class(**params)

        for i, (cv_train_data, cv_val_data) in enumerate(cross_validation_method(train_data, cross_validation_parameters)):
            cv_model_dir_name = '{}_fold'.format(i)
            cv_model_path = os.path.join(model_dir, cv_model_dir_name)
            definitions.make_dirs(cv_model_path)

            logging.info(cv_model_path)

            train_score, val_score, test_score = __cv_iteration(
                model,
                cv_model_path,
                cv_train_data,
                cv_val_data,
                test_data,
                get_generator,
                using_metrics,
            )
            cv_scores[i] = [train_score[using_metrics[0]], val_score[using_metrics[0]], test_score[using_metrics[0]]]

        model_parameters = model.get_params()
        mean_cv_scores = cv_scores.mean(axis=0)
        cv_scores = np.append(cv_scores, [mean_cv_scores], axis=0)

        __save_cross_validation_scores(cv_scores, cv_splits_count, model_dir)

        write_message = dict(
            params=model_parameters,
            scores=dict(
                train=mean_cv_scores[0],
                val=mean_cv_scores[1],
                test=mean_cv_scores[2],
            ),
            model_path=model_dir,
        )

        search_results.append(write_message)

        with open(search_results_file, 'w') as file:
            json.dump(search_results, file)
            logging.info('overwrite model parameters file ({})'.format(search_results_file))


def __restore_search_state(parameters_generator, search_results_file):
    search_results = []
    if os.path.exists(search_results_file):
        with open(search_results_file) as json_file:
            search_results = json.load(json_file)

    passed_parameters_count = len(search_results)

    for i, params in enumerate(parameters_generator):
        if i == passed_parameters_count - 1:
            break

    return parameters_generator, search_results


def __create_model_directory(models_dir) -> str:
    inner_dirs = list(definitions.get_inner_dirs(models_dir))
    folder_name = 'cv_{}'.format(len(inner_dirs))

    model_dir = os.path.join(models_dir, folder_name)
    definitions.make_dirs(model_dir)

    return model_dir


def __cv_iteration(model, cv_model_path, cv_train_data, cv_val_data, test_data, get_x_y_method, using_metrics):
    train_X, train_y = get_x_y_method(cv_train_data)
    val_X, val_y = get_x_y_method(cv_val_data)
    test_X, test_y = get_x_y_method(test_data)

    loss_history_file = os.path.join(cv_model_path, 'loss_history')
    model_checkpoint_file = os.path.join(cv_model_path, 'model.checkpoint')
    cv_tensorboard_dir = os.path.join(cv_model_path, 'tensorboard_log')

    __reset_weights(model)
    model.fit(
        (train_X, train_y),
        val_data=(val_X, val_y),
        test_data=(test_X, test_y),
        loss_history_file_name=loss_history_file,
        model_checkpoint_file_name=model_checkpoint_file,
        tensorboard_log_dir=cv_tensorboard_dir,
    )

    model_file = os.path.join(cv_model_path, 'model')
    model.save_model(model_file)
    logging.info('model was saved at {}'.format(model_file))

    train_score = model.score(train_X, train_y, using_metrics)
    val_score = model.score(val_X, val_y, using_metrics)
    test_score = model.score(test_X, test_y, using_metrics)

    return train_score, val_score, test_score


def __reset_weights(model):
    session = K.get_session()
    tf.set_random_seed(definitions.sklearn_seed)
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)


def __save_cross_validation_scores(cv_scores, cv_splits_count, model_path):
    df = pd.DataFrame(data=cv_scores, columns=['train', 'val', 'test'])
    titles = ['cv_{}'.format(cv_index) for cv_index in range(cv_splits_count)]
    titles.append('mean')
    df['title'] = titles
    cv_scores_file = os.path.join(model_path, 'cv_scores.csv')
    df.to_csv(cv_scores_file, index=False)

    logging.info('cross_validation scores was saved at {}'.format(cv_scores_file))


def demo():
    processing_sequence = {
        'load_test_data': dict(
            features_count=1000,
            rows_count=None,
        ),
        'filter_data': dict(
            filtered_column='Tissue',
            using_values='Whole blood',
        ),
        'apply_logarithm': dict(
            shift=3,
        ),
        'normalization': dict(
            method='series',
        ),
    }

    from module.data_processing.processing_conveyor import processing_conveyor
    data = processing_conveyor(processing_sequence)
    print(data.processed_data[data.best_genes])

    train_data, test_data = get_train_test(data.processed_data)

    from definitions import sklearn_seed, MODELS_DIR, make_dirs
    cross_validation_parameters = dict(
        n_splits=5,
        random_state=sklearn_seed,
        shuffle=True,
    )

    cross_validation_method = choose_cross_validation(method='custom')
    get_x_y_method = lambda x: (x[data.best_genes], x['Age'])

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
    epochs_count = 1,
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
        use_early_stopping=(True,)
    )

    experiment_dir = os.path.join(MODELS_DIR, 'predict_age/mlp/test')
    make_dirs(experiment_dir)

    log_dir = os.path.join(experiment_dir, 'log')
    make_dirs(log_dir)

    log_name = 'log.log'
    logging.basicConfig(level=logging.DEBUG, filename=os.path.join(log_dir, log_name))
    logging.info('Read data')

    from module.models.mlp import MLP
    search_parameters(
        lambda **params: MLP(data.processing_sequence['load_test_data']['features_count'], **params),
        train_data,
        test_data,
        cross_validation_method,
        cross_validation_parameters,
        get_x_y_method,
        ['r2'],
        model_parameters_space,
        experiment_dir,
        'cv_results.json',
    )


if __name__ == '__main__':
    demo()
