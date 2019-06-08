import pytest
import numpy as np
from sklearn.model_selection import ParameterGrid, ParameterSampler, KFold
import definitions
from module.models.grid_search import load_best_model_parameters, saved_models_parameters_generator
from module.models.dae_keras import coeff_determination
from sklearn.metrics import r2_score


def test_load_grid_results():
    for parameters in saved_models_parameters_generator(
            model_directory=definitions.path_join(definitions.MODELS_DIR, 'predict_age/mlp'),
            results_file='cv_results.json'):

        print(parameters)


def test_load_best_model_parameters():
    parameters = load_best_model_parameters(
        model_directory=definitions.path_join(definitions.MODELS_DIR, 'predict_age/mlp'),
        results_file='cv_results.json',
    )

    print(parameters)


def test_parameter_sampler():
    search_method = lambda kwargs: ParameterSampler(kwargs, n_iter=5)

    grid = search_method(dict(
        a=[1, 2, 3],
        b=[4, 5, 6]
    ))

    for params in grid:
        print(params)


def test_r2_coeff():
    from keras import backend as K

    a = np.array([
        [1.2, 2.6, 3.5],
        [4.4, 5.8, 6.7],
    ])

    b = np.array([
        [1, 2, 3],
        [4, 5, 6],
    ])

    result = r2_score(b, a)
    print(result)



    a = K.variable([
        [1.2, 2.6, 3.5],
        [4.4, 5.8, 6.7],
    ])

    b = K.variable([
        [1, 2, 3],
        [4, 5, 6],
    ])

    result = coeff_determination(b, a)
    print(K.eval(result))

