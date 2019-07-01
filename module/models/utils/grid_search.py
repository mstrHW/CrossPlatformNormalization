from sklearn.model_selection import ParameterGrid, ParameterSampler, KFold
import os
import numpy as np
import json
import pandas as pd
import logging
import definitions
from module.data_processing.data_processing import get_train_test
from module.models.utils import utils


def create_model_directory(models_dir) -> str:
    inner_dirs = list(definitions.get_inner_dirs(models_dir))
    folder_name = 'cv_{}'.format(len(inner_dirs))

    model_dir = os.path.join(models_dir, folder_name)
    definitions.make_dirs(model_dir)

    return model_dir


def k_fold_splits(train_data, cross_validation_parameters):
    k_fold = KFold(**cross_validation_parameters)

    for train_indexes, val_indexes in k_fold.split(train_data[0]):
        train_X = train_data[0][train_indexes]
        val_X = train_data[0][val_indexes]

        train_y = train_data[1][train_indexes]
        val_y = train_data[1][val_indexes]

        yield (train_X, train_y), (val_X, val_y)


def separate_labels_splits(train_data, cross_validation_parameters):
    geos = train_data['GEO'].value_counts().keys()
    splits_count = cross_validation_parameters['n_splits']

    if splits_count > len(geos):
        raise ValueError

    for i, val_geo_name in enumerate(geos):
        if i < splits_count:
            cv_val_mask = train_data['GEO'] == val_geo_name

            cv_train_data = train_data[~cv_val_mask]
            cv_val_data = train_data[cv_val_mask]

            yield cv_train_data, cv_val_data


def k_fold_splits_generator(train_data, cross_validation_parameters):
    k_fold = KFold(**cross_validation_parameters)

    for train_indexes, val_indexes in k_fold.split(train_data):
        cv_train_data = train_data.iloc[train_indexes]
        cv_val_data = train_data.iloc[val_indexes]

        yield cv_train_data, cv_val_data


def save_cross_validation_scores(cv_scores, cv_splits_count, model_path):
    df = pd.DataFrame(data=cv_scores, columns=['train', 'val', 'test'])
    titles = ['cv_{}'.format(cv_index) for cv_index in range(cv_splits_count)]
    titles.append('mean')
    df['title'] = titles
    cv_scores_file = os.path.join(model_path, 'cv_scores.csv')
    df.to_csv(cv_scores_file, index=False)

    logging.info('cross_validation scores was saved at {}'.format(cv_scores_file))


def restore_search_state(parameters_generator, search_results_file):
    search_results = []
    if os.path.exists(search_results_file):
        with open(search_results_file) as json_file:
            search_results = json.load(json_file)

    passed_parameters_count = len(search_results)

    for i, params in enumerate(parameters_generator):
        if i == passed_parameters_count - 1:
            break

    return parameters_generator, search_results


def cv_iteration(model, cv_model_path, cv_train_data, cv_val_data, test_data, get_x_y_method, using_metrics):
    train_X, train_y = get_x_y_method(cv_train_data)
    val_X, val_y = get_x_y_method(cv_val_data)
    test_X, test_y = get_x_y_method(test_data)

    loss_history_file = os.path.join(cv_model_path, 'loss_history')
    model_checkpoint_file = os.path.join(cv_model_path, 'model.checkpoint')
    cv_tensorboard_dir = os.path.join(cv_model_path, 'tensorboard_log')

    utils.reset_weights(model)
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


def search_parameters(model_class,
                      train_data,
                      test_data,
                      cross_validation_method,
                      cross_validation_parameters,
                      get_x_y_method,
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
    parameters_generator, search_results = restore_search_state(parameters_generator, search_results_file)

    cv_splits_count = cross_validation_parameters['n_splits']

    trained_models_dir = os.path.join(experiment_dir, 'trained_models')
    definitions.make_dirs(trained_models_dir)
    logging.info('trained models dir created at path : {}'.format(trained_models_dir))

    for params in parameters_generator:
        print(params)
        logging.info('current parameters : {}'.format(params))

        model_dir = create_model_directory(trained_models_dir)
        logging.info('model directory : {}'.format(model_dir))

        cv_scores = np.zeros((cv_splits_count, 3))
        model = model_class(**params)

        for i, (cv_train_data, cv_val_data) in enumerate(cross_validation_method(train_data, cross_validation_parameters)):
            cv_model_dir_name = '{}_fold'.format(i)
            cv_model_path = os.path.join(model_dir, cv_model_dir_name)
            definitions.make_dirs(cv_model_path)

            logging.info(cv_model_path)

            train_score, val_score, test_score = cv_iteration(
                model,
                cv_model_path,
                cv_train_data,
                cv_val_data,
                test_data,
                get_x_y_method,
                using_metrics,
            )
            cv_scores[i] = [train_score['r2'], val_score['r2'], test_score['r2']]

            cv_scores[i] = [0, 0, 0]

        model_parameters = model.get_params()
        mean_cv_scores = cv_scores.mean(axis=0)
        cv_scores = np.append(cv_scores, [mean_cv_scores], axis=0)

        save_cross_validation_scores(cv_scores, cv_splits_count, model_dir)

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


def saved_models_parameters_generator(model_directory, results_file):
    cv_results_file = os.path.join(model_directory, results_file)

    with open(cv_results_file) as json_file:
        data = json.load(json_file)

        for i, node in enumerate(data):
            yield node['params'], node['scores'], node['model_path']


def all_models_predict(model_class,
                       data,
                       model_directory,
                       results_file,
                       cross_validation_parameters,
                       predicts_file='predicts.json',
                       ):

    train_data, test_data = data

    cv_splits_count = cross_validation_parameters['n_splits']

    for (parameters, scores, model_files) in saved_models_parameters_generator(model_directory, results_file):

        for cv_train_data, cv_val_data in k_fold_splits(train_data, cv_splits_count):
            model = model_class(**parameters)
            model.load_model(model_files['model_file'])

            train_y_pred = model.predict(cv_train_data[0])
            val_y_pred = model.predict(cv_val_data[0])
            test_y_pred = model.predict(test_data[0])

            write_message = dict(
                train=train_y_pred,
                val=val_y_pred,
                test=test_y_pred,
            )

            predicts_file = os.path.join(model_files['model_dir'], predicts_file)

            save_predicts(predicts_file, write_message)


def save_predicts(predicts_file, write_message):
    with open(predicts_file, 'w') as file:
        json.dump(write_message, file)
        logging.info('predicts was saved at {}'.format(predicts_file))


def load_best_model_parameters(model_directory, results_file):
    cv_scores = []
    data = []

    for (parameters, scores, model_path) in saved_models_parameters_generator(model_directory, results_file):
        data.append([parameters, scores, model_path])
        cv_scores.append([scores['train'], scores['val'], scores['test']])

    cv_scores = np.array(cv_scores)
    best_model_index = cv_scores[:, 2].argmin()

    best_parameters = data[best_model_index][0]
    best_scores = data[best_model_index][1]
    best_model_path = data[best_model_index][2]

    return best_parameters, best_scores, best_model_path


def choose_cross_validation(method='default'):
    cross_validation = None
    if method == 'default':
        cross_validation = k_fold_splits_generator
    elif method == 'custom':
        cross_validation = separate_labels_splits
    return cross_validation


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

    from module.data_processing.data_generating_cases import processing_conveyor
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
