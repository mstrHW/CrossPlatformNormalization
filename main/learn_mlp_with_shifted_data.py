import numpy as np
import keras
from tqdm import tqdm
import definitions
from module.data_processing.noising_methods import gaussian_noise
import pandas as pd


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


class NewNoisedDataGenerator(keras.utils.Sequence):

    def __init__(self,
                 reference_batch_data,
                 corrupt_data,
                 gene_names,
                 mode,
                 batch_size=32,
                 shuffle=True,
                 noising_method='shift',
                 shift_probability=1.0,
                 generating_times=10,
                 ):

        np.random.seed(definitions.np_seed)
        self.calculate_reference_batch_distribution(reference_batch_data[gene_names])
        self.calculate_corrupt_batches_distribution(corrupt_data, gene_names)
        self.calculate_distance_distribution()

        self.corrupt_batch_names = corrupt_data['GEO'].unique()
        self.mode = mode
        self.noising_method = noising_method
        self.shift_probability = shift_probability
        self.generating_times = generating_times
        self.gene_names = gene_names

        if self.mode == 'train':
            self.__generate_noise = lambda x_shape, mean, std: gaussian_noise(x_shape, mean, std)
            self.__data = self.reference_batch_data
        elif self.mode == 'test':
            self.__generate_noise = lambda x_shape, mean, std: - gaussian_noise(x_shape, mean, std)
            self.__data = self.corrupt_data
        else:
            self.__data = pd.concat([self.corrupt_data, self.reference_batch_data])

        #         self.__data = self.__dataself.gene_names

        self.data_count = self.__data.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.on_epoch_end()

    def calculate_reference_batch_distribution(self, reference_batch_data):
        answer = reference_batch_data.apply(lambda x: get_distribution_params(x), axis=0)
        self.reference_batch_data = reference_batch_data
        self.reference_batch_distribution = np.moveaxis(answer.values, 0, 1)

    def calculate_corrupt_batches_distribution(self, corrupt_data, gene_names):
        self.corrupt_data = corrupt_data
        print(corrupt_data.shape)
        self.corrupt_distribution_params = calculate_gene_distribution_params(corrupt_data,
                                                                              corrupt_data['GEO'].unique(), gene_names)

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
        return int(self.generating_times)

    def __getitem__(self, index):
        indexes = np.random.choice(self.data_count, self.batch_size)

        X = self.__data[self.gene_names].iloc[indexes].values
        y = self.__data['Age'].iloc[indexes]

        corrupt_probability = 0.8
        flag = np.random.choice(2, 1, p=[1 - corrupt_probability, corrupt_probability])[0]

        corrupt_X = X
        if flag:
            corrupt_X = self.data_generation(X)

        #         return corrupt_X[self.gene_names], X
        #         X['GEO'] = [self.corrupt_batch_names[corrupt_batch] for i in range(X.shape[0])]
        return corrupt_X, y

    def data_generation(self, X):
        corrupt_batch = random_batch(self.corrupt_batch_count)

        for i in range(X.shape[1]):
            cutted_X = X[:, i]

            flag = np.random.choice(2, 1, p=[1 - self.shift_probability, self.shift_probability])[0]

            if self.noising_method == 'noise':
                mean_ = 0.
                std_ = 1.
                cutted_X = cutted_X + gaussian_noise(cutted_X.shape, mean_, std_)
            elif self.noising_method == 'shift':
                mean_, var_, std_ = self.distance[corrupt_batch, i]

                if flag == 1:
                    cutted_X = cutted_X + self.__generate_noise(cutted_X.shape, mean_, std_)

            X[:, i] = cutted_X

        return X

import pickle
import sys
sys.path.append("../")
from definitions import *

data_params = dict(
    features_count=1000,
    rows_count=None,
    filtered_column='Tissue',
    using_values='Whole blood',
    target_column='Age',
    normalize=True,
    use_generator=False,
    noising_method=None,
    batch_size=128,
)

from module.data_processing.data_processing import load_data, filter_data, get_train_test
from sklearn.preprocessing import MinMaxScaler, minmax_scale


data, best_genes = load_data(data_params['features_count'])
data = filter_data(data, data_params['filtered_column'], data_params['using_values'])

train_data, test_data = get_train_test(data)

ref_batch_name = 'GSE33828'

normalized_ref_data = train_data[train_data['GEO'] == ref_batch_name].copy()
normalized_train_data = train_data[train_data['GEO'] != ref_batch_name].copy()
normalized_test_data = test_data.copy()


def shift_with_log(X):
    return np.log(X + 1)

scaler = None

if data_params['normalize']:
    # scaler = MinMaxScaler(data[best_genes])
    columns = best_genes.tolist()
    columns.append('GEO')

    my_preprocessing = lambda x: minmax_scale(shift_with_log(x))
    cutted = normalized_ref_data[columns]
    normalized_ref_data[best_genes] = cutted.groupby('GEO').transform(lambda x: my_preprocessing(x))

    cutted = normalized_train_data[columns]
    normalized_train_data[best_genes] = cutted.groupby('GEO').transform(lambda x: my_preprocessing(x))

    cutted = normalized_test_data[columns]
    normalized_test_data[best_genes] = cutted.groupby('GEO').transform(lambda x: my_preprocessing(x))


print(normalized_train_data.shape)
print(normalized_train_data[best_genes].shape)


# train_generator = NewNoisedDataGenerator(
#     normalized_ref_data,
#     normalized_train_data,
#     best_genes,
#     'test',
#     batch_size=data_params['batch_size'],
#     noising_method=data_params['noising_method'],
#     # shift_probability=0.5,
# )
#
# test_generator = NewNoisedDataGenerator(
#     normalized_ref_data,
#     normalized_test_data,
#     best_genes,
#     'test',
#     batch_size=data_params['batch_size'],
#     noising_method=None,
#     # shift_probability=0.5,
# )


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


import json
from module.models.mlp import MLP


def load_best_model(model_class, model_directory, results_file):
    cv_results_file = os.path.join(model_directory, results_file)

    with open(cv_results_file) as json_file:
        data = json.load(json_file)
        cv_scores = np.zeros(shape=(len(data), 3))

        for i, node in enumerate(data):
            scores = node['scores']
            cv_scores[i] = [scores['train'], scores['val'], scores['test']]

        best_model_index = cv_scores[:, 2].argmin()

        print(best_model_index)
        print(data[best_model_index])

        best_parameters = data[best_model_index]['params']
        best_scores = data[best_model_index]['scores']
        best_model_path = data[best_model_index]['model_path']

        model = model_class(**best_parameters)

        return model, best_model_path


best_mlp_model, best_model_path = load_best_model(
    MLP,
    os.path.join(MODELS_DIR, 'predict_age/mlp'),
    'cv_results.json',
)

cv_model_path = '/home/aonishchuk/projects/CPN2/test_models/learn_mlp_with_shifted_data/normalized'
make_dirs(cv_model_path)

learning_params = dict(
    use_early_stopping=True,
    loss_history_file_name=os.path.join(cv_model_path, 'loss_history'),
    model_checkpoint_file_name=os.path.join(cv_model_path, 'model.checkpoint'),
    tensorboard_log_dir=os.path.join(cv_model_path, 'tensorboard_log'),
    # generator=True,
)


best_mlp_model.fit(
    (normalized_train_data[best_genes], normalized_train_data['Age']),
    (normalized_test_data[best_genes], normalized_test_data['Age']),
    **learning_params
)
#
# best_mlp_model.fit_generator(
#     train_generator,
#     (normalized_test_data[best_genes], normalized_test_data['Age']),
#     **learning_params
# )


best_mlp_model.save_model(os.path.join(cv_model_path, 'model'))

train_pred = best_mlp_model.predict(normalized_train_data[best_genes])
print(train_pred.shape)
print(normalized_train_data['Age'].shape)
test_pred = best_mlp_model.predict(normalized_test_data[best_genes])
#
# if data_params['normalize']:
#     train_pred = scaler.inverse_transform(train_pred)
#     test_pred = scaler.inverse_transform(test_pred)


from sklearn.metrics import mean_absolute_error, r2_score

columns = 'Age'
train_score = mean_absolute_error(normalized_train_data[columns], train_pred), r2_score(normalized_train_data[columns], train_pred)
test_score = mean_absolute_error(normalized_test_data[columns], test_pred), r2_score(normalized_test_data[columns], test_pred)

write_message = dict(
    train_results=train_score,
    test_results=test_score,
)


import logging
logging.basicConfig(level=logging.DEBUG, filename=r'log.log')

results_file = os.path.join(cv_model_path, 'results')

with open(results_file, 'w') as file:
    json.dump(write_message, file)
    logging.info('overwrite model parameters file ({})'.format(results_file))
print(train_score, test_score)