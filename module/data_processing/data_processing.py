import logging
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from module.data_processing.data_generator import NoisedDataGenerator
from module.data_processing.read_data import read_csv, read_genes
from definitions import *


logging.basicConfig(level=logging.DEBUG)


def load_data(features_count, tissues=None, rows_count=None):
    logging.debug('Read files')
    data = read_csv(illu_file, rows_count)
    best_genes = read_genes(best_genes_file)[:features_count]

    if tissues is None:
        tissues = data['Tissue'].unique()

    if type(tissues) is str:
        tissues = [tissues]

    cutted_data = data[data['Tissue'].isin(tissues)]
    cutted_data = cutted_data[pd.to_numeric(cutted_data['Age'], errors='coerce').notnull()]
    cutted_data['Age'] = cutted_data['Age'].fillna(cutted_data['Age'].mean())

    return cutted_data, best_genes


def get_train_test(data):
    train_mask = data['Training'] == 'Y'
    train_data = data[train_mask]
    test_data = data[~train_mask]

    return train_data, test_data


def get_x_y(data, best_genes):
    X = data[best_genes].astype('float').values
    y = data['Age'].astype('float').values

    return X, y


def fit_scaler(data, best_genes):
    scaler = MinMaxScaler()
    scaler.fit(data[best_genes])
    return scaler


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


def main(need_noise, **data_params):
    data, best_genes = load_data(data_params['features_count'], data_params['tissues'], data_params['rows_count'])

    if data_params['normalize']:
        scaler = fit_scaler(data, best_genes)
        data[best_genes] = scaler.transform(data[best_genes])

    train_data, test_data = get_train_test(data)

    if need_noise:
        train_data_ = make_shifted_data_generator(train_data, best_genes, data_params['batch_size'], data_params['noising_method'])
    else:
        train_data_ = get_x_y(train_data, best_genes)

    if need_noise:
        test_data_ = make_shifted_data_generator(test_data, best_genes, data_params['batch_size'], 'None')
    else:
        test_data_ = get_x_y(test_data, best_genes)

    return train_data_, test_data_


if __name__ == '__main__':
    main()
