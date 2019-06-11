from sklearn.model_selection import ParameterGrid, ParameterSampler, KFold
import os
import numpy as np
import json
import pandas as pd
import logging
import definitions
from module.data_processing.data_processing import make_shifted_data_generator, get_train_test, fit_scaler
from module.data_processing.new_data_generator import NewNoisedDataGenerator


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


def k_fold_splits(train_data, cross_validation_parameters):
    k_fold = KFold(**cross_validation_parameters)

    for train_indexes, val_indexes in k_fold.split(train_data[0]):
        train_X = train_data[0][train_indexes]
        val_X = train_data[0][val_indexes]

        train_y = train_data[1][train_indexes]
        val_y = train_data[1][val_indexes]

        yield (train_X, train_y), (val_X, val_y)


def k_fold_splits_generator(train_data, cross_validation_parameters):
    k_fold = KFold(**cross_validation_parameters)

    for train_indexes, val_indexes in k_fold.split(train_data):
        train_X = train_data.iloc[train_indexes]
        val_X = train_data.iloc[val_indexes]

        yield (train_X, val_X)


def save_cross_validation_scores(cv_scores, cv_splits_count, model_path):
    df = pd.DataFrame(data=cv_scores, columns=['train', 'val', 'test'])
    titles = ['cv_{}'.format(cv_index) for cv_index in range(cv_splits_count)]
    titles.append('mean')
    df['title'] = titles
    cv_scores_file = os.path.join(model_path, 'cv_scores.csv')
    df.to_csv(cv_scores_file, index=False)

    logging.info('cross_validation scores was saved at {}'.format(cv_scores_file))


def restore_search_state(parameters_generator, search_results_file,):
    search_results = []
    if os.path.exists(search_results_file):
        with open(search_results_file) as json_file:
            search_results = json.load(json_file)

    passed_parameters_count = len(search_results)

    for i, params in parameters_generator:
        if i == passed_parameters_count - 1:
            break

    return parameters_generator, search_results


def reset_weights(model):
    from keras import backend as K
    import tensorflow as tf
    session = K.get_session()
    tf.set_random_seed(definitions.sklearn_seed)
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)


def cv_iteration(k_fold, model, model_dir, train_data, test_data, learning_parameters, using_metrics):
    # k_fold = k_fold_splits(train_data, cross_validation_parameters)
    for i, (cv_train_data, cv_val_data) in enumerate(k_fold):
        cv_model_dir_name = '{}_fold'.format(i)
        logging.info(cv_model_dir_name)

        cv_model_path = os.path.join(model_dir, cv_model_dir_name)
        definitions.make_dirs(cv_model_path)

        loss_history_file = os.path.join(cv_model_path, 'loss_history')
        model_checkpoint_file = os.path.join(cv_model_path, 'model.checkpoint')
        tensorboard_log_dir = os.path.join(cv_model_path, 'tensorboard_log')
        model_file = os.path.join(cv_model_path, 'model')


        reset_weights(model)
        model.fit(
            cv_train_data,
            val_data=cv_val_data,
            test_data=test_data,
            use_early_stopping=learning_parameters['use_early_stopping'],
            loss_history_file_name=loss_history_file,
            model_checkpoint_file_name=model_checkpoint_file,
            tensorboard_log_dir=tensorboard_log_dir,
        )

        model.save_model(model_file)
        logging.info('model was saved at {}'.format(model_file))

        model_parameters = model.get_params()

        # using_metrics = ['r2', 'mae']

        train_score = model.score(*cv_train_data, using_metrics)['mae']
        val_score = model.score(*cv_val_data, using_metrics)['mae']
        test_score = model.score(*test_data, using_metrics)['mae']
    return train_score, val_score, test_score


def search_parameters(model_class,
                      train_data,
                      test_data,
                      model_parameters_space,
                      learning_parameters,
                      cross_validation_parameters,
                      experiment_dir,
                      results_file,
                      search_method_name='grid',
                      random_n_iter=150,
                      ):

    logging.info('Start search method : {}'.format(search_method_name))
    search_method = ParameterGrid
    if search_method_name == 'random':
        search_method = lambda kwargs: ParameterSampler(kwargs,
                                                        n_iter=random_n_iter,
                                                        random_state=definitions.sklearn_seed)

    parameters_generator = search_method(model_parameters_space)

    search_results_file = os.path.join(experiment_dir, results_file)
    parameters_generator, search_results = restore_search_state(parameters_generator, search_results_file)

    cv_splits_count = cross_validation_parameters['n_splits']
    tensorboard_log_dir = os.path.join(experiment_dir, 'tensorboard_log')

    for params in parameters_generator:

        logging.info('current parameters : {}'.format(params))

        model_dir = create_model_directory(experiment_dir)

        cv_scores = np.zeros((cv_splits_count, 3))
        model_parameters = []

        for i, (cv_train_data, cv_val_data) in enumerate(k_fold_splits(train_data, cross_validation_parameters)):
            cv_model_dir_name = '{}_fold'.format(i)
            logging.info(cv_model_dir_name)

            cv_model_path = os.path.join(model_dir, cv_model_dir_name)
            definitions.make_dirs(cv_model_path)

            loss_history_file = os.path.join(cv_model_path, 'loss_history')
            model_checkpoint_file = os.path.join(cv_model_path, 'model.checkpoint')

            model = model_class(**params)
            model.fit(
                cv_train_data,
                val_data=cv_val_data,
                test_data=test_data,
                use_early_stopping=learning_parameters['use_early_stopping'],
                loss_history_file_name=loss_history_file,
                model_checkpoint_file_name=model_checkpoint_file,
                tensorboard_log_dir=tensorboard_log_dir,
            )

            model_file = os.path.join(cv_model_path, 'model')
            model.save_model(model_file)
            logging.info('model was saved at {}'.format(model_file))

            model_parameters = model.get_params()

            using_metrics = ['r2', 'mae']

            train_score = model.score(*cv_train_data, using_metrics)['mae']
            val_score = model.score(*cv_val_data, using_metrics)['mae']
            test_score = model.score(*test_data, using_metrics)['mae']

            cv_scores[i] = [train_score, val_score, test_score]

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


def search_parameters_generator(model_class,
                                ref_batch_name,
                                data,
                                best_genes,
                                data_params,
                                model_parameters_space,
                                learning_parameters,
                                cross_validation_parameters,
                                model_directory,
                                results_file,
                                search_method_name='grid',
                                random_n_iter=150,
                                ):

    logging.info('Start search method : {}'.format(search_method_name))
    search_method = ParameterGrid
    if search_method_name == 'random':
        search_method = lambda kwargs: ParameterSampler(kwargs,
                                                        n_iter=random_n_iter,
                                                        random_state=definitions.sklearn_seed)

    train_data, test_data = get_train_test(data)

    normalized_ref_data = train_data[train_data['GEO'] == ref_batch_name].copy()
    normalized_train_data = train_data[train_data['GEO'] != ref_batch_name].copy()
    normalized_test_data = test_data.copy()

    scaler = None
    if data_params['normalize']:
        scaler = fit_scaler(data, best_genes)
        normalized_ref_data[best_genes] = scaler.transform(normalized_ref_data[best_genes])
        normalized_train_data[best_genes] = scaler.transform(normalized_train_data[best_genes])
        normalized_test_data[best_genes] = scaler.transform(normalized_test_data[best_genes])

    grid = search_method(model_parameters_space)

    best_parameters = None
    best_scores = np.array([np.inf, np.inf, np.inf])
    best_model_path = None

    search_results_file = os.path.join(model_directory, results_file)
    search_results = []
    if os.path.exists(search_results_file):
        with open(search_results_file) as json_file:
            search_results = json.load(json_file)

    search_results_count = len(search_results)

    cv_splits_count = cross_validation_parameters['n_splits']

    for i, params in enumerate(grid):

        logging.info('current parameters : {}'.format(params))

        if i < search_results_count:
            i += 1
            continue

        model_dir_name = get_unique_folder_name(model_directory)
        model_path = os.path.join(model_directory, model_dir_name)
        definitions.make_dirs(model_path)

        cv_scores = np.zeros((cv_splits_count, 3))
        model_parameters = []

        for i, (cv_train_data, cv_val_data) in enumerate(k_fold_splits_generator(normalized_train_data, cross_validation_parameters)):
            logging.info('{}_fold'.format(i))

            cv_model_dir_name = str(i)
            cv_model_path = os.path.join(model_path, cv_model_dir_name)
            definitions.make_dirs(cv_model_path)

            train_generator = NewNoisedDataGenerator(
                normalized_ref_data,
                cv_train_data,
                best_genes,
                'train',
                batch_size=data_params['batch_size'],
                noising_method=data_params['noising_method'],
            )

            val_generator = NewNoisedDataGenerator(
                normalized_ref_data,
                cv_val_data,
                best_genes,
                'test',
                batch_size=data_params['batch_size'],
                noising_method=data_params['noising_method'],
            )

            test_generator = NewNoisedDataGenerator(
                normalized_ref_data,
                test_data,
                best_genes,
                'test',
                batch_size=data_params['batch_size'],
                noising_method=data_params['noising_method'],
            )

            model = model_class(**params)
            model.fit_generator(
                train_generator,
                test_data=val_generator,
                use_early_stopping=learning_parameters['use_early_stopping'],
                loss_history_file_name=os.path.join(cv_model_path, 'loss_history'),
                model_checkpoint_file_name=os.path.join(cv_model_path, 'model.checkpoint'),
                tensorboard_log_dir=os.path.join(cv_model_path, 'tensorboard_log'),
            )

            model_file = os.path.join(cv_model_path, 'model')
            model.save_model(model_file)
            logging.info('model was saved at {}'.format(model_file))

            model_parameters = model.get_params()

            using_metrics = ['r2', 'mae']

            train_score = model.score_generator(train_generator, scaler, using_metrics)['r2']
            val_score = model.score_generator(val_generator, scaler, using_metrics)['r2']
            test_score = model.score_generator(test_generator, scaler, using_metrics)['r2']

            cv_scores[i] = [train_score, val_score, test_score]

        mean_cv_scores = cv_scores.mean(axis=0)
        cv_scores = np.append(cv_scores, [mean_cv_scores], axis=0)

        save_cross_validation_scores(cv_scores, cv_splits_count, model_path)

        if mean_cv_scores[1] < best_scores[1]:
            best_scores = mean_cv_scores
            best_parameters = model_parameters
            best_model_path = model_path

        write_message = dict(
            params=model_parameters,
            scores=dict(
                train=mean_cv_scores[0],
                val=mean_cv_scores[1],
                test=mean_cv_scores[2],
            ),
            model_path=model_path,
        )

        search_results.append(write_message)
        search_results_file = os.path.join(model_directory, results_file)

        with open(search_results_file, 'w') as file:
            json.dump(search_results, file)
            logging.info('overwrite model parameters file ({})'.format(search_results_file))

    return best_scores, best_parameters, best_model_path


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
