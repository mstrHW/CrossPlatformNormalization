import json
import pandas as pd

from definitions import *


def load_best_model(model_class, model_directory, results_file, best_on='test', condition='min'):
    cv_results_file = os.path.join(model_directory, results_file)

    condition_index = 0
    if best_on == 'val':
        condition_index = 1
    elif best_on == 'test':
        condition_index = 2

    condition_lambda = lambda x: x.argmin()
    if condition == 'max':
        condition_lambda = lambda x: x.argmax()

    model = None
    with open(cv_results_file) as json_file:
        data = json.load(json_file)
        cv_scores = np.zeros(shape=(len(data), 3))

        for i, node in enumerate(data):
            scores = node['scores']
            cv_scores[i] = [scores['train'], scores['val'], scores['test']]

        best_model_index = condition_lambda(cv_scores[:, condition_index])

        print(best_model_index)
        print(data[best_model_index])

        best_parameters = data[best_model_index]['params']
        best_scores = data[best_model_index]['scores']
        best_model_path = data[best_model_index]['model_path']

        cv_scores_file = os.path.join(best_model_path, 'cv_scores.csv')
        cv_scores_data = pd.read_csv(cv_scores_file)

        best_model_index = condition_lambda(cv_scores_data[best_on].values)
        best_model_path = os.path.join(best_model_path, str(best_model_index))

        model = model_class(**best_parameters)
        model.load_model(os.path.join(best_model_path, 'model'))

    return model
