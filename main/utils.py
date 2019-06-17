import json

from definitions import *


def load_best_model(model_class, model_directory, results_file):
    cv_results_file = path_join(model_directory, results_file)

    with open(cv_results_file) as json_file:
        data = json.load(json_file)
        cv_scores = np.zeros(shape=(len(data), 3))

        for i, node in enumerate(data):
            scores = node['scores']
            cv_scores[i] = [scores['train'], scores['val'], scores['test']]

        best_model_index = cv_scores[:, 2].argmin()

        print(best_model_index)
        print(data[best_model_index])

        best_parameters = data[best_model_index]['params']
        best_scores = data[best_model_index]['scores']
        best_model_path = data[best_model_index]['model_path']

        model = model_class(**best_parameters)

        return model, best_model_path
