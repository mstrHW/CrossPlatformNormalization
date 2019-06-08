import logging

from module.models.mlp import MLP
from module.data_processing import data_processing
from module.models.grid_search import search_parameters
from module.models.utils import save_results, save_search_results
from definitions import *
import json
from imp import reload
reload(logging)


def main():
    logging.basicConfig(level=logging.DEBUG, filename=r'log.log')
    logging.debug('Read data')
    data_params = dict(
        features_count=1000,
        tissues=['Whole blood'],
        normalize=True,
        noising_method=None,
        batch_size=512,
        rows_count=None
    )

    train_data, test_data = data_processing.main(False, **data_params)

    logging.debug('Create model')

    model_params = dict(
        layers=(1500, 1200, 1000, 800, 700, 500, 256, 128, 1),
        activation='lrelu',
        drop_rate=0.5,
        regularizer_name='l1_l2',
        regularizer_param=1e-3,
        epochs_count=10,
        learning_rate=0.00001,
        loss='mae',
        batch_size=data_params['batch_size'],
        patience=200,
        optimizer_name='adam',
    )

    # layers = [
    #     (512, 256, 128),
    #     (768, 384, 192),
    #     (1024, 512, 256, 128),
    #     (1536, 768, 384, 192),
    #     (1024, 768, 512, 384, 128),
    #     (1536, 1024, 768, 512, 192),
    # ]
    # activation = ['elu', 'lrelu', 'prelu']
    # optimizer = ['adam, rmsprop']

    model = MLP(features_count=data_params['features_count'], **model_params)


    learning_params = dict(
        use_early_stopping=True,
        loss_history_file_name=None,
        model_checkpoint_file_name=None,
    )

    logging.debug('Fit model')
    model.fit(*train_data, test_data=test_data, **learning_params)

    # train_score = model.calculate_score(train_data[0], train_data[1])
    # test_score = model.calculate_score(test_data[0], test_data[1])
    #
    # save_results('MLP', data_params, model_params, train_score, test_score)


def search_model_parameters():
    np.random.seed(np_seed)
    tf.random.set_random_seed(np_seed)

    experiment_path = os.path.join(MODELS_DIR, 'predict_age/mlp')
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    logging.basicConfig(level=logging.DEBUG, filename=os.path.join(experiment_path, 'log.log'))

    logging.info('Read data')
    data_params = dict(
        features_count=1000,
        tissues=['Whole blood'],
        normalize=True,
        noising_method=None,
        batch_size=256,
        rows_count=None,
    )

    experiment_meta_params_file = os.path.join(experiment_path, 'experiment_meta_parameters.json')
    with open(experiment_meta_params_file, 'w') as file:
        write_message = dict(
            np_seed=np_seed,
            sklearn_seed=sklearn_seed,
            data_params=data_params,
        )
        json.dump(write_message, file)
        logging.info('experiment meta parameters was saved at file {}'.format(experiment_meta_params_file))

    train_data, test_data = data_processing.main(False, **data_params)

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

    learning_params = dict(
        use_early_stopping=True,
        # loss_history_file_name=,
        # model_checkpoint_file_name=None,
    )

    search_parameters(
        lambda **kwargs: MLP(features_count=1000, **kwargs),
        (train_data, test_data),
        model_parameters_space,
        learning_parameters=learning_params,
        cross_validation_parameters=dict(n_splits=5, random_state=sklearn_seed),
        model_directory=experiment_path,
        results_file='cv_results.json',
        search_method_name='random',
        random_n_iter=150,
    )

    # save_search_results(*gs_results, 'MLP_GS')


if __name__ == '__main__':
    # main()
    search_model_parameters()
