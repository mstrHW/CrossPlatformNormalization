import os
import json
import logging
import numpy as np
import pandas as pd


def saved_models_parameters_generator(model_directory, results_file):
    cv_results_file = os.path.join(model_directory, results_file)

    with open(cv_results_file) as json_file:
        data = json.load(json_file)

        for i, node in enumerate(data):
            yield node['params'], node['scores'], node['model_path']


def load_best_model_parameters(model_directory, results_file, best_on='test', condition='min'):
    cv_scores = []
    data = []

    condition_index = 0
    if best_on == 'val':
        condition_index = 1
    elif best_on == 'test':
        condition_index = 2

    condition_lambda = lambda x: x.argmin()
    if condition == 'max':
        condition_lambda = lambda x: x.argmax()

    for (parameters, scores, model_path) in saved_models_parameters_generator(model_directory, results_file):
        data.append([parameters, scores, model_path])
        cv_scores.append([scores['train'], scores['val'], scores['test']])

    cv_scores = np.array(cv_scores)
    best_model_index = condition_lambda(cv_scores[:, condition_index])

    best_parameters = data[best_model_index][0]
    best_scores = data[best_model_index][1]
    best_model_path = data[best_model_index][2]

    cv_scores_file = os.path.join(best_model_path, 'cv_scores.csv')
    cv_scores_data = pd.read_csv(cv_scores_file)

    best_model_index = condition_lambda(cv_scores_data[best_on].values)
    best_fold_path = os.path.join(best_model_path, '{}_fold'.format(best_model_index))

    if not os.path.exists(best_fold_path):  # for older version of experiment structure
        best_fold_path = os.path.join(best_model_path, str(best_model_index))

    return best_parameters, best_scores, best_model_path, best_fold_path


def all_models_predict(model_class,
                       train_data,
                       test_data,
                       model_directory,
                       results_file,
                       cross_validation_method,
                       cross_validation_parameters,
                       predicts_file='predicts.json',
                       ):

    for (parameters, scores, model_files) in saved_models_parameters_generator(model_directory, results_file):

        for cv_train_data, cv_val_data in cross_validation_method(train_data, cross_validation_parameters):
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

            _predicts_file = os.path.join(model_files['model_dir'], predicts_file)

            __save_predicts(_predicts_file, write_message)


def __save_predicts(predicts_file, write_message):
    with open(predicts_file, 'w') as file:
        json.dump(write_message, file)
        logging.info('predicts was saved at {}'.format(predicts_file))


if __name__ == '__main__':
    from definitions import *
    from module.models.mlp import MLP

    # best_parameters, best_scores, best_model_path, best_fold_path = load_best_model_parameters(
    #     path_join(MODELS_DIR, 'predict_age/mlp'),
    #     'cv_results.json',
    # )
    #
    # model = MLP(**best_parameters)
    # model.load_model(path_join(best_fold_path, 'model'))

