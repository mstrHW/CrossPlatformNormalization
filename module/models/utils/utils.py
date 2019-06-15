import csv
import pandas as pd
from keras import backend as K
import tensorflow as tf

from definitions import *


def save_results(model_name, data_params, model_params, learning_params, train_score, test_score):
    fields = ['model_name', 'data_params', 'model_params', 'learning_params', 'train_score', 'test_score']
    results_file = os.path.join(MODELS_DIR, '{}_results.csv'.format(model_name))
    file_exists = os.path.exists(results_file)

    with open(results_file, 'a') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(fields)
        writer.writerow([model_name, data_params, model_params, learning_params, train_score, test_score])


def save_search_results(cv_results, best_params, best_score, title):
    results_df = pd.DataFrame.from_dict(cv_results)
    results_file = os.path.join(MODELS_DIR, '{}_results.csv'.format(title))
    results_df.to_csv(results_file)

    best_model_file = os.path.join(MODELS_DIR, '{}_best_model.csv'.format(title))
    fields = ['best_params', 'best_score']
    file_exists = os.path.exists(best_model_file)

    with open(best_model_file, 'a') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(fields)
        writer.writerow([best_params, best_score])


def reset_weights(model):
    session = K.get_session()
    tf.set_random_seed(sklearn_seed)
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)
