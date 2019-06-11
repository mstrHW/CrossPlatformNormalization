import logging
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from module.data_processing.data_generator import NoisedDataGenerator
from module.data_processing.read_data import read_csv, read_genes
from definitions import *


logging.basicConfig(level=logging.DEBUG)


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


def make_shifted_data_generator(data, best_genes, batch_size, noising_method='shift'):
    np.random.seed(np_seed)
    value_counts = data['GEO'].value_counts()
    unique_geos = data['GEO'].unique()

    max_count_class = value_counts.index.tolist()[0]
    training_generator = NoisedDataGenerator(
        data,
        max_count_class,
        unique_geos,
        best_genes,
        batch_size=batch_size,
        noising_method=noising_method,
    )
    return training_generator
