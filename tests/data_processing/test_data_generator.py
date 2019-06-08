import pytest
from module.data_processing.data_generator import NoisedDataGenerator
from definitions import *
from module.data_processing.read_data import read_csv, read_genes
from module.data_processing.data_processing import load_data, get_train_test, make_shifted_data_generator


def test_data_generation():
    assert False


def test_using_data_generator():
    data = read_csv(illu_file, None)
    using_geos = data['GEO'].unique()
    using_genes = read_genes(best_genes_file)[:1000]
    ref_geo_name = data['GEO'].value_counts().index.values[0]
    print(ref_geo_name)
    generator = NoisedDataGenerator(data, ref_geo_name, using_geos, using_genes, batch_size=256)


def test_fixed_seed():
    # np.random.seed(np_seed)
    data_params = dict(
        features_count=1000,
        tissues=['Whole blood'],
        normalize=False,
        noising_method='shift',
        batch_size=128,
        rows_count=None,
    )

    data, best_genes = load_data(data_params['features_count'], data_params['tissues'], data_params['rows_count'])
    train_data, test_data = get_train_test(data)

    generator = make_shifted_data_generator(train_data, best_genes, data_params['batch_size'], data_params['noising_method'])

    count = 0
    data = []
    for i in generator:
        count += 1
        data.append(i)
        if count > 1:
            break

    generator = make_shifted_data_generator(train_data, best_genes, data_params['batch_size'], data_params['noising_method'])

    count = 0
    data2 = []
    for i in generator:
        count += 1
        data2.append(i)
        if count > 1:
            break

    print(data)
    print(data2)
