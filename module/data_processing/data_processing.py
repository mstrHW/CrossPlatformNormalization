import logging
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, minmax_scale

from module.data_processing.noising_methods import gaussian_noise
from module.data_processing.read_data import read_csv, read_genes
from definitions import *


def load_data(features_count, rows_count=None):
    logging.debug('Read files')
    data = read_csv(illu_file, rows_count)
    best_genes = read_genes(best_genes_file)[:features_count]

    return data, best_genes


def filter_data(data, filtered_column, using_values):
    if type(using_values) is str:
        using_values = [using_values]

    filtered_data = data[data[filtered_column].isin(using_values)]

    return filtered_data


def get_X_y(data, using_genes, target_column=None):
    X = data[using_genes].astype('float').values

    numeric_targets = data[pd.to_numeric(data[target_column], errors='coerce').notnull()]
    y = numeric_targets[target_column].astype('float').values

    return X, y


def get_train_test(data):
    train_mask = data['Training'] == 'Y'
    train_data = data[train_mask]
    test_data = data[~train_mask]

    return train_data, test_data


def fit_scaler(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    return scaler


def normalize_by_series(data, best_genes):
    columns = best_genes.tolist()
    columns.append('GEO')

    cutted_data = data.loc[:, columns]
    data.loc[:, best_genes] = cutted_data.groupby('GEO').transform(lambda x: minmax_scale(x))

    return data


def get_batches(data, batch_size):
    for i in range(0, data.shape[0], batch_size):
        yield data[i:i + batch_size]


def add_gaussian_noise(data, batch_size, noise_probability_for_gene):
    for batch in get_batches(data, batch_size):
        batch_shape = (batch.shape[0], data.shape[1])

        noising_flags = np.random.choice(
            2,
            batch_shape,
            p=[1-noise_probability_for_gene, noise_probability_for_gene],
        )

        noising_flags = np.array(noising_flags, dtype=bool)

        noise = gaussian_noise(batch_shape, 0.5, 0.5)
        batch = batch.where(noising_flags, batch + noise)
        yield batch


def apply_log(data, shift=0.):
    return np.log(data + shift)


def revert_log(data, shift=0.):
    return np.exp(data) - shift
