import numpy as np
from sklearn.preprocessing import minmax_scale, MinMaxScaler

from module.data_processing.noising_methods import gaussian_noise


def filter_data(data, filtered_column, using_values):
    if type(using_values) is str:
        using_values = [using_values]

    filtered_data = data[data[filtered_column].isin(using_values)]

    return filtered_data


def get_train_test(data):
    train_mask = data['Training'] == 'Y'
    train_data = data[train_mask]
    test_data = data[~train_mask]

    return train_data, test_data


def normalize_by_series(data, best_genes):
    columns = best_genes.tolist()
    columns.append('GEO')

    cutted_data = data.loc[:, columns]
    data.loc[:, best_genes] = cutted_data.groupby('GEO').transform(lambda x: minmax_scale(x))

    return data


def apply_log(data, shift=0.):
    return np.log(data + shift)


def revert_log(data, shift=0.):
    return np.exp(data) - shift


def get_batches(data, batch_size):
    for i in range(0, data.shape[0], batch_size):
        yield data[i:i + batch_size]


def add_gaussian_noise(data, noise_probability_for_gene):
    batch_shape = (data.shape[0], data.shape[1])

    noising_flags = np.random.choice(
        2,
        batch_shape,
        p=[1-noise_probability_for_gene, noise_probability_for_gene],
    )

    noising_flags = np.array(noising_flags, dtype=bool)

    noise = gaussian_noise(batch_shape, 0.5, 0.5)
    data = np.where(noising_flags, data, data + noise)
    return data
