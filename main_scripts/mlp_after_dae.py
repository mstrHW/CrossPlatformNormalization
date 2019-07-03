import sys
sys.path.append("../")
import logging
import os
from sklearn.metrics import mean_absolute_error, r2_score
import json
import argparse
from imp import reload

from definitions import *
from module.models.dae import DenoisingAutoencoder
from module.data_processing.processing_conveyor import ProcessingConveyor
from module.data_processing.data_processing import get_train_test, get_batches
from module.models.mlp import MLP
from main_scripts.utils import load_best_model


def main(args):
    reload(logging)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device_number

    np.random.seed(np_seed)
    tf.set_random_seed(np_seed)

    log_file = path_join(args.experiment_dir, 'log.log')
    logging.basicConfig(level=logging.DEBUG, filename=log_file)
    logging.debug('Read data')

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

    models_dir = args.dae_models_dir
    best_dae_model = load_best_model(
        DenoisingAutoencoder,
        models_dir,
        'cv_results.json',
        condition='max',
    )

    model_path = args.experiment_dir
    make_dirs(model_path)

    learning_params = dict(
        loss_history_file_name=os.path.join(model_path, 'loss_history'),
        model_checkpoint_file_name=os.path.join(model_path, 'model.checkpoint'),
        tensorboard_log_dir=os.path.join(model_path, 'tensorboard_log'),
    )

    target_column = 'Age'

    (train_X, train_y) = train_data[best_genes], train_data[target_column]
    (test_X, test_y) = test_data[best_genes], test_data[target_column]
    train_X_ = best_dae_model.predict(train_X)
    test_X_ = best_dae_model.predict(test_X)

    train_data_ = (train_X, train_y)
    test_data_ = (test_X_, test_y)

    models_dir = args.mlp_models_dir
    best_mlp_model = load_best_model(
        MLP,
        models_dir,
        'cv_results.json',
    )

    best_mlp_model.fit(
        train_data_,
        test_data_,
        **learning_params,
    )

    best_mlp_model.save_model(os.path.join(model_path, 'model'))

    train_pred = best_mlp_model.predict(train_data[best_genes])
    test_pred = best_mlp_model.predict(test_data[best_genes])

    train_score = dict(
        mae=mean_absolute_error(train_data[target_column], train_pred),
        r2=r2_score(train_data[target_column], train_pred),
    )

    test_score = dict(
        mae=mean_absolute_error(test_data[target_column], test_pred),
        r2=r2_score(test_data[target_column], test_pred),
    )

    write_message = dict(
        train_results=train_score,
        test_results=test_score,
    )

    results_file = path_join(args.experiment_dir, args.results_file_name)

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
        "--dae_models_dir",
        type=str,
        help="increase output verbosity",
    )

    parser.add_argument(
        "--mlp_models_dir",
        type=str,
        help="increase output verbosity",
    )

    parser.add_argument(
        "--results_file_name",
        type=str,
        default='results.json',
        help="increase output verbosity",
    )

    parser.add_argument(
        "--cuda_device_number",
        type=str,
        default='0',
        help="increase output verbosity",
    )
    args = parser.parse_args()
    main(args)
