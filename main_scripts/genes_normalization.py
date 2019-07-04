import logging
import os
import json
from imp import reload
import argparse
import sys
sys.path.append("../")
import keras
import math

from module.models.dae import DenoisingAutoencoder
from module.data_processing.data_processing import get_train_test
from module.models.utils.cross_validation import choose_cross_validation
from module.models.utils.grid_search import search_parameters
from module.data_processing.processing_conveyor import ProcessingConveyor
from definitions import *

from module.data_processing.distance_noise_generation import DistanceNoiseGenerator


class DataGenerator(keras.utils.Sequence):
    def __init__(self, ref_data, data, best_genes, mode, noise_probability, batch_size=128):
        if mode == 'train':
            self.data = ref_data
        else:
            self.data = data

        self.data_count = self.data.shape[0]
        self.best_genes = best_genes
        self.batch_size = batch_size
        self.noise_generator = DistanceNoiseGenerator(
            ref_data,
            data,
            best_genes,
            mode,
            noise_probability,
        )
        self.on_epoch_end()

    def __len__(self):
        return int(math.ceil(self.data_count / self.batch_size))

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size

        batch = self.data.iloc[start_index: end_index].copy()
        batch.loc[:, self.best_genes] = self.noise_generator.data_generation(batch[self.best_genes].values)
        X = self.data.iloc[start_index: end_index][self.best_genes].values
        y = batch[self.best_genes].values

        return X, y

    def on_epoch_end(self):
        pass


def search_model_parameters(args):
    reload(logging)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device_number

    np.random.seed(np_seed)
    tf.set_random_seed(np_seed)

    experiment_dir = args.experiment_dir
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    log_file = path_join(experiment_dir, 'log.log')
    logging.basicConfig(level=logging.DEBUG, filename=log_file)

    logging.info('Read data')

    processing_sequence = {
        'load_data': dict(
            features_count=1000,
            rows_count=None,
        ),
        'filter_data': dict(
            filtered_column='Tissue',
            using_values='Whole blood',
        ),
        'normalization': dict(
            method='series',
        ),
    }

    data_wrapper = ProcessingConveyor(processing_sequence)
    train_data, test_data = get_train_test(data_wrapper.processed_data)

    ref_batch_name = train_data['GEO'].value_counts().keys()[0]
    ref_mask = train_data['GEO'] == ref_batch_name

    best_genes = data_wrapper.best_genes

    ref_data = train_data[ref_mask]
    train_data = train_data[~ref_mask]

    cross_validation_method_name = 'default'
    cross_validation_parameters = dict(
        n_splits=5,
        random_state=sklearn_seed,
        shuffle=True,
    )

    cross_validation_method = choose_cross_validation(cross_validation_method_name)

    # columns = data_wrapper.best_genes.tolist()
    # columns.append('GEO')

    create_train_generator = lambda data: DataGenerator(
        ref_data,
        data,
        best_genes,
        'train',
        0.5,
    )

    create_test_generator = lambda data: DataGenerator(
        ref_data,
        data,
        best_genes,
        'test',
        0.5,
    )

    experiment_meta_params_file = os.path.join(experiment_dir, 'experiment_meta_parameters.json')
    with open(experiment_meta_params_file, 'w') as file:
        write_message = dict(
            np_seed=np_seed,
            sklearn_seed=sklearn_seed,
        )
        json.dump(write_message, file)
        logging.info('experiment meta parameters was saved at file {}'.format(experiment_meta_params_file))

    data_parameters_file = os.path.join(experiment_dir, 'data_parameters.json')
    with open(data_parameters_file, 'w') as file:
        write_message = dict(
            data_processing=processing_sequence,
            cross_validation_method=cross_validation_method_name,
            cross_validation=cross_validation_parameters,
        )
        json.dump(write_message, file)
        logging.info('data processing parameters was saved at file {}'.format(data_parameters_file))

    logging.info('Start grid search')

    layers = [
        (
            (1000, 512, 256, 128, 64),
            (128, 256, 512, 1000)
        ),
        (
            (1000, 512, 256, 64),
            (256, 512, 1000)
        ),
        (
            (1000, 384, 64),
            (384, 1000)
        ),
    ]

    activation = ['elu', 'lrelu', 'prelu']
    regularization_param = [10 ** -i for i in range(3, 7)]
    epochs_count = 1,
    loss = 'mae',
    optimizer = ['adam', 'rmsprop'] #, 'eve's

    model_parameters_space = dict(
        layers=layers,
        activation=activation,
        regularizer_param=regularization_param,
        epochs_count=epochs_count,
        loss=loss,
        optimizer_name=optimizer,
    )

    search_parameters(
        lambda **params: DenoisingAutoencoder(
            data_wrapper.processing_sequence['load_data']['features_count'],
            **params,
        ),
        train_data,
        test_data,
        cross_validation_method,
        cross_validation_parameters,
        create_train_generator,
        create_test_generator,
        ['r2'],
        model_parameters_space,
        experiment_dir,
        args.cv_results_file_name,
        args.search_method,
        args.n_iters,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_dir",
        type=str,
        help="increase output verbosity",
    )

    parser.add_argument(
        "--cv_results_file_name",
        type=str,
        default='cv_results.json',
        help="increase output verbosity")

    parser.add_argument(
        "--search_method",
        type=str,
        default='random',
        help="increase output verbosity",

    )
    parser.add_argument(
        "--n_iters",
        type=int,
        default=100,
        help="increase output verbosity",
    )

    parser.add_argument(
        "--cuda_device_number",
        type=str,
        default='0',
        help="increase output verbosity",
    )
    args = parser.parse_args()
    search_model_parameters(args)
