import logging
import json
from imp import reload
import os
import argparse
import sys
sys.path.append('../')

from module.data_processing.processing_conveyor import ProcessingConveyor
from module.data_processing.data_processing import *
from module.models.dae_with_predictor import DAEwithPredictor
from module.models.mlp import MLP
from definitions import *


def search_model_parameters(args):
    reload(logging)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device_number

    np.random.seed(np_seed)
    tf.set_random_seed(np_seed)

    experiment_dir = args.experiment_dir
    make_dirs(experiment_dir)

    log_file = path_join(args.experiment_dir, 'log.log')
    logging.basicConfig(level=logging.DEBUG, filename=log_file)

    logging.info('Read data')

    processing_sequence = {
        'load_test_data': dict(
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
    best_genes = data_wrapper.best_genes

    logging.info('Start grid search')

    model = DAEwithPredictor(
        data_wrapper.processing_sequence['load_test_data']['features_count'],
        layers=[
            (1000, 384, 64),
            (384, 1000)
        ],
        activation='elu',
        regularizer_name='l1_l2',
        regularizer_param=1e-3,
        epochs_count=2000,
        learning_rate=0.0001,
        loss='mae',
        patience=200,
        optimizer_name='adam',
    )

    experiment_meta_params_file = os.path.join(experiment_dir, 'experiment_meta_parameters.json')
    with open(experiment_meta_params_file, 'w') as file:
        write_message = dict(
            np_seed=np_seed,
            sklearn_seed=sklearn_seed,
        )
        json.dump(write_message, file)
        logging.info('experiment meta parameters was saved at file {}'.format(experiment_meta_params_file))

    model_dir = os.path.join(experiment_dir, 'dae')
    make_dirs(model_dir)

    loss_history_file = os.path.join(model_dir, 'loss_history')
    model_checkpoint_file = os.path.join(model_dir, 'model.checkpoint')
    tensorboard_dir = os.path.join(model_dir, 'tensorboard_log')
    model_file = os.path.join(model_dir, 'model')

    if os.path.exists(model_file):
        model.load_model(model_file)
    else:
        model.fit(
            train_data,
            best_genes,
            test_data,
            loss_history_file_name=loss_history_file,
            model_checkpoint_file_name=model_checkpoint_file,
            tensorboard_log_dir=tensorboard_dir,
        )

        model.save_model(model_file)

    train_score = model.score(
        *train_data,
        ['r2', 'mae'],
    )

    test_score = model.score(
        *test_data,
        ['r2', 'mae'],
    )

    write_message = dict(
        train_results=train_score,
        test_results=test_score,
    )

    results_file = os.path.join(model_dir, args.results_file_name)

    with open(results_file, 'w') as file:
        json.dump(write_message, file)
        logging.info('overwrite model parameters file ({})'.format(results_file))

    decoded_train_X = model.predict(train_data[0])[0]
    decoded_test_X = model.predict(test_data[0])[0]

    model_params = dict(
        layers=(1500, 800, 700, 500, 128, 1),
        activation='elu',
        drop_rate=0.5,
        regularizer_name='l1_l2',
        regularizer_param=1e-3,
        epochs_count=2000,
        loss='mae',
        patience=200,
        optimizer_name='adam',
    )

    model = MLP(
        data_wrapper.processing_sequence['load_test_data']['features_count'],
        **model_params,
    )

    logging.debug('Fit MLP model')

    model_dir = os.path.join(experiment_dir, 'mlp')
    make_dirs(model_dir)

    loss_history_file = os.path.join(model_dir, 'loss_history')
    model_checkpoint_file = os.path.join(model_dir, 'model.checkpoint')
    tensorboard_dir = os.path.join(model_dir, 'tensorboard_log')
    model_file = os.path.join(model_dir, 'model')

    learning_params = dict(
        loss_history_file_name=loss_history_file,
        model_checkpoint_file_name=model_checkpoint_file,
        tensorboard_log_dir=tensorboard_dir,
    )

    if os.path.exists(model_file):
        model.load_model(model_file)
    else:
        model.fit(
            train_data,
            (decoded_test_X, test_data[1]),
            **learning_params,
        )

        model.save_model(model_file)

    train_score = model.score_sklearn(
        train_data,
        ['r2', 'mae'],
    )

    test_score = model.score_sklearn(
        (decoded_test_X, test_data[1]),
        ['r2', 'mae'],
    )

    write_message = dict(
        train_results=train_score,
        test_results=test_score,
    )

    results_file = os.path.join(model_dir, args.results_file_name)

    with open(results_file, 'w') as file:
        json.dump(write_message, file)
        logging.info('overwrite model parameters file ({})'.format(results_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_dir",
        type=str,
        help="increase output verbosity",
    )

    parser.add_argument(
        "--results_file_name",
        type=str,
        default='results.json',
        help="increase output verbosity")

    parser.add_argument(
        "--cuda_device_number",
        type=str,
        default='0',
        help="increase output verbosity",
    )
    args = parser.parse_args()
    search_model_parameters(args)

