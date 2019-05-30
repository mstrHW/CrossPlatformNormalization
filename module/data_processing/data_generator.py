import numpy as np
import keras
from tqdm import tqdm

from module.data_processing.noising_methods import add_gaussian_noise


def random_for_each_gene(corrupt_batch_count, genes_count):
    return np.random.randint(corrupt_batch_count, size=(genes_count,))


def random_batch(corrupt_batch_count):
    return np.random.randint(corrupt_batch_count)


def get_distribution_params(gene):
    mean_ = gene.mean()
    var_ = gene.var(ddof=0)
    std_ = gene.std(ddof=0)

    return np.array([mean_, var_, std_])


def calculate_gene_distribution_params(data, geo_names, gene_names):
    geo_count = len(geo_names)
    gene_count = len(gene_names)
    batch_distribution_params = np.zeros((geo_count, gene_count, 3))

    for i, geo in tqdm(enumerate(geo_names)):
        batch_genes = data[(data['GEO'] == geo)][gene_names]
        answer = batch_genes.apply(lambda x: get_distribution_params(x), axis=0)
        answer = np.moveaxis(answer.values, 0, 1)
        batch_distribution_params[i] = answer

    return batch_distribution_params


class NoisedDataGenerator(keras.utils.Sequence):
    def __init__(self, data, ref_batch_name, geo_names, gene_names, batch_size=32, shuffle=True,
                 noising_method='shift'):
        self.batch_distribution_params = calculate_gene_distribution_params(data, geo_names, gene_names)
        print(self.batch_distribution_params.shape)

        self.noising_method = noising_method

        self.geo_count = len(geo_names)
        self.gene_count = len(gene_names)

        self.geo_names = geo_names
        self.gene_names = gene_names

        print(self.geo_names)

        distance = np.zeros((self.geo_count, self.gene_count, 2))

        reference_batch_index = self.geo_names.tolist().index(ref_batch_name)
        reference_batch_params = self.batch_distribution_params[reference_batch_index]

        for corrupt_batch_index, geo in enumerate(self.geo_names):
            if corrupt_batch_index == reference_batch_index:
                continue

            corrupt_batch_params = self.batch_distribution_params[corrupt_batch_index]
            corrupt_batch_params[:, 0] = corrupt_batch_params[:, 0] - reference_batch_params[:, 0]
            corrupt_batch_params[:, 1] = corrupt_batch_params[:, 1] + reference_batch_params[:, 1]
            corrupt_batch_params[:, 2] = np.sqrt(corrupt_batch_params[:, 1])

            distance[corrupt_batch_index] = corrupt_batch_params[:, [0, 2]]

        distance = np.delete(distance, reference_batch_index, axis=0)

        self.distance = distance
        self.corrupt_batch_count = self.distance.shape[0]

        self.batch_size = batch_size

        self.data = data[data['GEO'] == ref_batch_name][self.gene_names]
        self.data_count = self.data.shape[0]
        self.ref_batch_name = ref_batch_name
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.data_count / self.batch_size))

    def __getitem__(self, index):
        indexes = np.random.choice(self.data_count, self.batch_size)
        X = self.data.iloc[indexes]
        X = X.values

        corrupt_X = self.data_generation(X)

        return X, corrupt_X

    def data_generation(self, X):
        corrupt_batch = random_batch(self.corrupt_batch_count)
        corrupt_X = X.copy()

        for i in range(corrupt_X.shape[1]):
            cutted_X = corrupt_X[:, i]

            if self.noising_method == 'noise':
                mean_ = 0.
                std_ = 1.
                cutted_X = add_gaussian_noise(cutted_X, mean_, std_)
            elif self.noising_method == 'shifted':
                mean_, std_ = self.distance[corrupt_batch, i]
                cutted_X = add_gaussian_noise(cutted_X, mean_, std_)

            corrupt_X[:, i] = cutted_X

        return corrupt_X

    def sample_data(self):
        reference_batch_index = self.geo_names.tolist().index(self.ref_batch_name)
        corrupt_batch_index = random_batch(self.corrupt_batch_count)

        reference_batch_params = self.batch_distribution_params[reference_batch_index]
        corrupt_batch_params = self.batch_distribution_params[corrupt_batch_index]

        X = np.zeros((self.batch_size * 2, self.gene_count))
        for i in range(self.gene_count):
            mean, _, std = reference_batch_params[i]
            X[:self.batch_size, i] = np.random.normal(mean, std, self.batch_size)

        for i in range(self.gene_count):
            mean, _, std = corrupt_batch_params[i]
            X[self.batch_size:, i] = np.random.normal(mean, std, self.batch_size)

        corrupt_X = X.copy()
        corrupt_X[:self.batch_size] = self.data_generation(corrupt_X[:self.batch_size])
        return X, corrupt_X
