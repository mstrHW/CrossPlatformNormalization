import numpy as np
from tqdm import tqdm

import definitions
from module.data_processing.noising_methods import gaussian_noise


class DistanceNoiseGenerator(object):

    def __init__(self,
                 reference_batch_data,
                 corrupt_data,
                 gene_names,
                 mode,
                 noise_probability_for_gene,
                 ):

        np.random.seed(definitions.np_seed)

        self.corrupt_batch_names = corrupt_data['GEO'].unique()
        self.corrupt_batches_count = len(self.corrupt_batch_names)

        reference_batch_distribution = self.__calculate_reference_batch_distribution(
            reference_batch_data[gene_names]
        )

        corrupt_distribution_params = self.__calculate_gene_distribution_params(
            corrupt_data,
            self.corrupt_batch_names,
            gene_names,
        )

        self.distance = self.__calculate_distance_distribution(
            reference_batch_distribution,
            corrupt_distribution_params,
        )

        self.mode = mode
        self.shift_probability = noise_probability_for_gene

        if self.mode == 'train':
            self.__generate_noise = lambda x_shape, mean, std: gaussian_noise(x_shape, mean, std)
        else:
            self.__generate_noise = lambda x_shape, mean, std: - gaussian_noise(x_shape, mean, std)

    def data_generation(self, X):
        selected_batches = np.random.choice(self.corrupt_batches_count, X.shape)
        selected_batches = selected_batches.shape[1] * selected_batches + np.arange(selected_batches.shape[1])  # for broadcasting

        means = np.take(self.distance[0], selected_batches)
        stds = np.take(self.distance[1], selected_batches)

        selected_genes = np.random.choice(2, X.shape, p=[1 - self.shift_probability, self.shift_probability])
        selected_genes = np.array(selected_genes, dtype=bool)

        noise = self.__generate_noise(X.shape, means, stds)
        X = np.where(selected_genes, X, X + noise)

        return X

    @staticmethod
    def __calculate_reference_batch_distribution(reference_batch_data):
        answer = reference_batch_data.apply(lambda x: DistanceNoiseGenerator.__get_distribution_params(x), axis=0)
        return np.moveaxis(answer.values, 0, 1)

    @staticmethod
    def __get_distribution_params(gene):
        mean_ = gene.mean()
        var_ = gene.var(ddof=0)
        std_ = gene.std(ddof=0)

        return np.array([mean_, var_, std_])

    @staticmethod
    def __calculate_gene_distribution_params(data, geo_names, gene_names):
        geo_count = len(geo_names)
        gene_count = len(gene_names)
        batch_distribution_params = np.zeros((geo_count, gene_count, 3))

        for i, geo in tqdm(enumerate(geo_names)):
            batch_genes = data[(data['GEO'] == geo)][gene_names]
            answer = batch_genes.apply(lambda x: DistanceNoiseGenerator.__get_distribution_params(x), axis=0)
            answer = np.moveaxis(answer.values, 0, 1)
            batch_distribution_params[i] = answer

        return batch_distribution_params

    @staticmethod
    def __calculate_distance(corrupt_batch_params, reference_batch_params):
        distance = np.zeros(shape=corrupt_batch_params.shape)
        distance[:, 0] = corrupt_batch_params[:, 0] - reference_batch_params[:, 0]
        distance[:, 1] = corrupt_batch_params[:, 1] + reference_batch_params[:, 1]
        distance[:, 2] = np.sqrt(distance[:, 1])

        return distance[:, 0], distance[:, 2]

    @staticmethod
    def __calculate_distance_distribution(reference_batch_distribution, corrupt_distribution_params):
        cdp_shape = corrupt_distribution_params.shape
        means = np.zeros(shape=(cdp_shape[0], cdp_shape[1]))
        stds = np.zeros(shape=(cdp_shape[0], cdp_shape[1]))

        for i, corrupt_batch_params in enumerate(corrupt_distribution_params):
            means[i], stds[i] = DistanceNoiseGenerator.__calculate_distance(corrupt_batch_params, reference_batch_distribution)

        return means, stds


if __name__ == '__main__':
    pass