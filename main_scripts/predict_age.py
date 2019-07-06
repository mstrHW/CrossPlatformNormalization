import logging
import json
from imp import reload
import os
import argparse
import sys
sys.path.append('../')
import keras
import math

from module.models.mlp import MLP
from module.models.utils.grid_search import search_parameters
from module.models.utils.cross_validation import choose_cross_validation
from module.data_processing.processing_conveyor import ProcessingConveyor
from module.data_processing.data_processing import get_train_test
from definitions import *


class DataGenerator(keras.utils.Sequence):
    def __init__(self, data, best_genes, batch_size=128):
        self.data = data
        self.data_count = data.shape[0]
        self.best_genes = best_genes
        self.batch_size = batch_size

        self.on_epoch_end()

    def __len__(self):
        return int(math.ceil(self.data_count / self.batch_size))

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size

        batch = self.data.iloc[start_index: end_index]
        X = batch[self.best_genes].values
        y = batch['Age'].values

        return X, y

    def on_epoch_end(self):
        pass


def search_model_parameters(args):
    reload(logging)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device_number

    np.random.seed(np_seed)
    tf.set_random_seed(np_seed)

    experiment_dir = args.experiment_dir
    make_dirs(experiment_dir)

    log_dir = os.path.join(experiment_dir, 'log')
    make_dirs(log_dir)

    log_name = 'log.log'
    log_file = path_join(log_dir, log_name)

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

    logging.info('Start grid search')

    layers = [
        (256, 128, 64, 1),
        (512, 256, 128, 1),
        (512, 384, 256, 128, 1),
        (768, 512, 384, 192, 1),
        (1024, 768, 512, 384, 128, 1),
        (1536, 1024, 768, 384, 192, 1),
    ]
    activation = ['elu', 'lrelu', 'prelu']
    dropout_rate = [0.25, 0.5, 0.75]
    regularization_param = [10 ** -i for i in range(3, 7)]
    epochs_count = 2000,
    loss = 'mae',
    optimizer = ['adam', 'rmsprop'] #, 'eve's

    model_parameters_space = dict(
        layers=layers,
        drop_rate=dropout_rate,
        activation=activation,
        regularizer_param=regularization_param,
        epochs_count=epochs_count,
        loss=loss,
        optimizer_name=optimizer,
    )

    cross_validation_method_name = 'custom'
    cross_validation_parameters = dict(
        n_splits=5,
        random_state=sklearn_seed,
        shuffle=True,
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
        logging.info('experiment meta parameters was saved at file {}'.format(data_parameters_file))

    cross_validation_method = choose_cross_validation(cross_validation_method_name)

    create_generator = lambda data: DataGenerator(data, data_wrapper.best_genes)

    search_parameters(
        lambda **params: MLP(data_wrapper.processing_sequence['load_data']['features_count'], **params),
        train_data,
        test_data,
        cross_validation_method,
        cross_validation_parameters,
        create_generator,
        create_generator,
        ['mae'],
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
        help="path to directory for current experiment trained models and results",
    )

    parser.add_argument(
        "--cv_results_file_name",
        type=str,
        default='cv_results.json',
        help="file name of search parameters results")

    parser.add_argument(
        "--search_method",
        type=str,
        default='random',
        help="choose 'random' or 'grid' search",

    )
    parser.add_argument(
        "--n_iters",
        type=int,
        default=100,
        help="number of search iterations for random search",
    )

    parser.add_argument(
        "--cuda_device_number",
        type=str,
        default='0',
        help="number of gpu for execute tensorflow",
    )
    args = parser.parse_args()
    search_model_parameters(args)
