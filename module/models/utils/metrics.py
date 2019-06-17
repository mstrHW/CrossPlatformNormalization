from keras import backend as K
from sklearn.metrics import mean_absolute_error, r2_score


def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred), axis=0)
    SS_tot = K.sum(K.square(y_true - K.mean(y_true, axis=0)), axis=0)
    r2_vect = (1 - SS_res/(SS_tot + K.epsilon()))
    result_mean = K.mean(r2_vect)
    return result_mean


__name_to_metric = {
    'r2': coeff_determination,
}


def make_metric(metric_name):
    try:
        metric = __name_to_metric[metric_name]
    except KeyError as e:
        raise ValueError('Undefined optimizer: {}'.format(e.args[0]))

    return metric


__name_to_sklearn_metric = {
    'r2': r2_score,
    'mae': mean_absolute_error,
}


def make_sklearn_metric(metric_name):
    try:
        metric = __name_to_sklearn_metric[metric_name]
    except KeyError as e:
        raise ValueError('Undefined optimizer: {}'.format(e.args[0]))

    return metric
