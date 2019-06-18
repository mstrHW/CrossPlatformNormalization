import numpy as np
import keras
from tqdm import tqdm
import definitions
from module.data_processing.noising_methods import gaussian_noise


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

    def __init__(self,
                 reference_batch_data,
                 corrupt_data,
                 gene_names,
                 mode,
                 noise_probability_for_gene,
                 batch_size=32,
                 shuffle=False,
                 ):

        np.random.seed(definitions.np_seed)
        self.calculate_reference_batch_distribution(reference_batch_data[gene_names])
        self.calculate_corrupt_batches_distribution(corrupt_data, gene_names)
        self.calculate_distance_distribution()

        self.corrupt_batch_names = corrupt_data['GEO'].unique()
        self.mode = mode
        self.shift_probability = noise_probability_for_gene

        if self.mode == 'train':
            self.__generate_noise = lambda x_shape, mean, std: gaussian_noise(x_shape, mean, std)
            self.__data = self.reference_batch_data
        else:
            self.__generate_noise = lambda x_shape, mean, std: - gaussian_noise(x_shape, mean, std)
            self.__data = self.corrupt_data

        self.__data = self.__data[gene_names]

        self.data_count = self.__data.shape[0]
        self.batch_size = batch_size
        self.indexes = np.arange(self.data_count)
        self.shuffle = shuffle

        self.on_epoch_end()

    def calculate_reference_batch_distribution(self, reference_batch_data):
        answer = reference_batch_data.apply(lambda x: get_distribution_params(x), axis=0)
        self.reference_batch_data = reference_batch_data
        self.reference_batch_distribution = np.moveaxis(answer.values, 0, 1)

    def calculate_corrupt_batches_distribution(self, corrupt_data, gene_names):
        self.corrupt_data = corrupt_data
        self.corrupt_distribution_params = calculate_gene_distribution_params(
            corrupt_data,
            corrupt_data['GEO'].unique(),
            gene_names,
        )

    def calculate_distance(self, corrupt_batch_params, reference_batch_params):
        distance = np.zeros(shape=corrupt_batch_params.shape)
        distance[:, 0] = corrupt_batch_params[:, 0] - reference_batch_params[:, 0]
        distance[:, 1] = corrupt_batch_params[:, 1] + reference_batch_params[:, 1]
        distance[:, 2] = np.sqrt(distance[:, 1])

        return distance

    def calculate_distance_distribution(self):
        distance = np.zeros(shape=self.corrupt_distribution_params.shape)

        for i, corrupt_batch_params in enumerate(self.corrupt_distribution_params):
            distance[i] = self.calculate_distance(corrupt_batch_params, self.reference_batch_distribution)

        self.distance = distance
        self.corrupt_batch_count = self.distance.shape[0]

    def __len__(self):
        return int(np.floor(self.data_count / self.batch_size))

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = np.random.choice(self.data_count, self.batch_size)
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        X = self.__data.iloc[indexes]

        corrupt_X = self.data_generation(X.values)

        if self.mode == 'train':
            answer = (corrupt_X, X)
        else:
            answer = (X, corrupt_X)

        return answer

    def data_generation(self, X):
        corrupt_batch = random_batch(self.corrupt_batch_count)

        flag = np.random.choice(2, X.shape[1], p=[1 - self.shift_probability, self.shift_probability])

        for i in range(X.shape[1]):
            cutted_X = X[:, i]

            mean_, var_, std_ = self.distance[corrupt_batch, i]

            if flag == 1:
                cutted_X = cutted_X + self.__generate_noise(cutted_X.shape, mean_, std_)

            X[:, i] = cutted_X

        return X

    def sample_data(self):
        '''
        Doesn't work
        :return:
        '''
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


if __name__ == '__main__':
    New