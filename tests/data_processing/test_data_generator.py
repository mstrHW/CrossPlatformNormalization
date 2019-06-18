import pytest
from sklearn.preprocessing import minmax_scale


from module.data_processing.data_generator import NoisedDataGenerator
from module.data_processing.noising_methods import gaussian_noise

from definitions import *
from module.data_processing.read_data import read_csv, read_genes
from module.data_processing.data_processing import load_data, get_train_test, make_shifted_data_generator, filter_data, fit_scaler


@pytest.mark.skip(reason="no way of currently testing this")
def test_data_generation():
    assert False


@pytest.mark.skip(reason="no way of currently testing this")
def test_using_data_generator():
    data = read_csv(illu_file, None)
    using_geos = data['GEO'].unique()
    using_genes = read_genes(best_genes_file)[:1000]
    ref_geo_name = data['GEO'].value_counts().index.values[0]
    print(ref_geo_name)
    generator = NoisedDataGenerator(data, ref_geo_name, using_geos, using_genes, batch_size=256)


@pytest.mark.skip(reason="no way of currently testing this")
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


@pytest.mark.skip(reason="no way of currently testing this")
def test_using_gauss_noise():
    noise_probability = 0.25
    batch_size = 128

    data, best_genes = load_data(1000, 100)
    data = filter_data(data, 'Tissue', 'Whole blood')

    scaler = fit_scaler(data[best_genes])
    train_data, test_data = get_train_test(data)

    train_data[best_genes] = scaler.transform(train_data[best_genes])
    # test_data[best_genes] = scaler.transform(test_data[best_genes])

    noised_batches_generator = add_gaussian_noise(train_data[best_genes], batch_size, noise_probability)
    batches_generator = get_batches(train_data[best_genes], batch_size)

    for batch, noised_batch in (batches_generator, noised_batches_generator):
        pass

    assert True


def shift_to_corrupt(ref_data, corrupt_data, best_genes, noise_probability, batch_size):
    noised_batches_generator = NoisedDataGenerator(
        ref_data,
        corrupt_data,
        best_genes,
        'train',
        noise_probability,
        batch_size=batch_size,
    )
    return noised_batches_generator


def shift_to_reference(corrupt_data, ref_data, best_genes, noise_probability, batch_size):
    noised_batches_generator = NoisedDataGenerator(
        ref_data,
        corrupt_data,
        best_genes,
        'test',
        noise_probability,
        batch_size=batch_size,
    )
    return noised_batches_generator


@pytest.mark.skip(reason="no way of currently testing this")
def test_using_distance_noise():
    noise_probability = 0.25
    batch_size = 128

    data, best_genes = load_data(1000, None)
    data = filter_data(data, 'Tissue', 'Whole blood')

    scaler = fit_scaler(data[best_genes])
    train_data, test_data = get_train_test(data)

    major_batch_name = train_data['GEO'].value_counts().keys()[0]

    ref_data = train_data[train_data['GEO'] == major_batch_name]
    _train_data = train_data[train_data['GEO'] != major_batch_name]
    _test_data = test_data.copy()

    _train_data.loc[:, best_genes] = scaler.transform(_train_data.loc[:, best_genes])

    train_noised_batches_generator = shift_to_corrupt(
        ref_data,
        _train_data,
        best_genes,
        noise_probability,
        batch_size=batch_size,
    )

    test_noised_batches_generator = shift_to_reference(
        ref_data,
        _test_data,
        best_genes,
        noise_probability,
        batch_size,
    )

    for batch in test_noised_batches_generator:
        pass

    assert True


def normalize_by_series(data, best_genes):
    columns = best_genes.tolist()
    columns.append('GEO')

    cutted_data = data.loc[:, columns]
    data.loc[:, best_genes] = cutted_data.groupby('GEO').transform(lambda x: minmax_scale(x))

    return data


@pytest.mark.skip(reason="no way of currently testing this")
def test_using_normalization_by_series():
    batch_size = 128

    data, best_genes = load_data(1000, None)
    data = filter_data(data, 'Tissue', 'Whole blood')

    train_data, test_data = get_train_test(data)

    normalized_data = normalize_by_series(train_data, best_genes)

    for batch in get_batches(normalized_data, batch_size):
        print(batch[best_genes])
        pass

    assert True


def apply_log(data, shift=0.):
    return np.log(data + shift)


def revert_log(data, shift=0.):
    return np.exp(data) - shift


@pytest.mark.skip(reason="no way of currently testing this")
def test_apply_log():
    batch_size = 128

    data, best_genes = load_data(1000, None)
    data = filter_data(data, 'Tissue', 'Whole blood')

    train_data, test_data = get_train_test(data)

    test_data.loc[:, best_genes] = apply_log(test_data.loc[:, best_genes], 3)

    import pandas as pd
    pd.options.mode.use_inf_as_na = True
    nans_summ = test_data[best_genes].isna().sum().sum()
    print(nans_summ)
    assert int(nans_summ) == 0

    normalized_data = normalize_by_series(test_data, best_genes)

    for batch in get_batches(normalized_data, batch_size):
        print(batch[best_genes])
        pass

    assert True


@pytest.mark.skip(reason="no way of currently testing this")
def test_distance_noise_with_log():
    batch_size = 128

    data, best_genes = load_data(1000, None)
    data = filter_data(data, 'Tissue', 'Whole blood')

    train_data, test_data = get_train_test(data)

    test_data.loc[:, best_genes] = apply_log(test_data.loc[:, best_genes], 3)

    import pandas as pd
    pd.options.mode.use_inf_as_na = True
    nans_summ = test_data[best_genes].isna().sum().sum()
    print(nans_summ)
    assert int(nans_summ) == 0

    normalized_data = normalize_by_series(test_data, best_genes)

    for batch in get_batches(normalized_data, batch_size):
        print(batch[best_genes])
        pass

    assert True


def data_generation(X, shift_probability, corrupt_batches_count):
    # corrupt_batch = random_batch(self.corrupt_batch_count)

    choosed_genes_flag = np.random.choice(2, X.shape, p=[1-shift_probability, shift_probability])
    choosed_genes_flag = np.array(choosed_genes_flag, dtype=bool)

    choosed_batches_flag = np.random.choice(corrupt_batches_count, X.shape)
    choosed_batches_flag = np.array(choosed_batches_flag, dtype=bool)

    print(choosed_genes_flag)

    means = dis

    means = [i for i in range(X.shape[1])]
    stds = [i/X.shape[1] for i in range(X.shape[1])]

    X[choosed_genes_flag] = X[choosed_genes_flag] + gaussian_noise(X.shape, means, stds)[choosed_genes_flag]

    # for i in range(X.shape[1]):
    #     cutted_X = X[:, i]
    #
    #     mean_, var_, std_ = self.distance[corrupt_batch, i]
    #
    #     if flag == 1:
    #         cutted_X = cutted_X + self.__generate_noise(cutted_X.shape, mean_, std_)
    #
    #     X[:, i] = cutted_X

    return X


def test_broadcast_add_distance_noise():
    np.random.seed(43)

    X = np.array([
        np.linspace(0, 1, 50),
        np.linspace(-1, 1, 50),
    ])
    print(X)

    _X = data_generation(X, 0.25)
    print(_X)

    assert True
