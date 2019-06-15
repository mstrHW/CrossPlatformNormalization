import logging

from module.models.dae import DenoisingAutoencoder
from module.data_processing.data_processing import load_data, get_train_test, filter_data
from module.models.utils.grid_search import search_parameters_generator
from sklearn.metrics import mean_absolute_error, r2_score
from definitions import *
import json
from imp import reload
reload(logging)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    logging.basicConfig(level=logging.DEBUG, filename=r'log.log')
    logging.debug('Read data')
    data_params = dict(
        features_count=1000,
        tissues=['Whole blood'],
        normalize=True,
        noising_method=None,
        batch_size=128,
        rows_count=None,
    )

    data, best_genes = load_data(data_params['features_count'], data_params['tissues'], data_params['rows_count'])
    train_data, test_data = get_train_test(data)

    normalized_train_data = train_data.copy()
    normalized_test_data = test_data.copy()

    scaler = fit_scaler(data, best_genes)
    if data_params['normalize']:
        normalized_train_data[best_genes] = scaler.transform(train_data[best_genes])
        normalized_test_data[best_genes] = scaler.transform(test_data[best_genes])


    # train_data_generator = make_shifted_data_generator(
    #     normalized_train_data,
    #     best_genes,
    #     data_params['batch_size'],
    #     data_params['noising_method'],
    # )
    #
    # test_data_generator = make_shifted_data_generator(
    #     normalized_test_data,
    #     best_genes,
    #     data_params['batch_size'],
    #     'None',
    # )

    logging.debug('Create model')

    model_params = dict(
        layers=[
            (1000, 384, 64),
            (384, 1000),
        ],
        activation='elu',
        # drop_rate=0.25,
        regularizer_name='l1_l2',
        regularizer_param=1e-5,
        epochs_count=2000,
        learning_rate=0.0001,
        loss='mae',
        # batch_size=data_params['batch_size'],
        patience=200,
        optimizer_name='adam',
    )

    model = DenoisingAutoencoder(features_count=data_params['features_count'], **model_params)

    cv_model_path = path_join(MODELS_DIR, 'genes_normalization/dae/norm_with_bn')
    make_dirs(cv_model_path)

    learning_params = dict(
        use_early_stopping=True,
        loss_history_file_name=os.path.join(cv_model_path, 'loss_history'),
        model_checkpoint_file_name=os.path.join(cv_model_path, 'model.checkpoint'),
        tensorboard_log_dir=os.path.join(cv_model_path, 'tensorboard_log'),
        # generator=True,
    )

    logging.debug('Fit model')
    # model.fit_generator(train_data_generator, test_data=test_data_generator, **learning_params)
    model.fit(
        (normalized_train_data[best_genes], normalized_train_data[best_genes]),
        test_data=(normalized_test_data[best_genes], normalized_test_data[best_genes]),
        **learning_params,
    )
    model.save_model(os.path.join(cv_model_path, 'model'))

    train_pred = model.predict(normalized_train_data[best_genes])
    test_pred = model.predict(normalized_test_data[best_genes])

    if data_params['normalize']:
        train_pred = scaler.inverse_transform(train_pred)
        test_pred = scaler.inverse_transform(test_pred)

    train_score = mean_absolute_error(train_data[best_genes], train_pred), r2_score(train_data[best_genes], train_pred)
    test_score = mean_absolute_error(test_data[best_genes], test_pred), r2_score(test_data[best_genes], test_pred)

    write_message = dict(
        train_results=train_score,
        test_results=test_score,
    )

    results_file = os.path.join(cv_model_path, 'results')
    with open(results_file, 'w') as file:
        json.dump(write_message, file)
        logging.info('overwrite model parameters file ({})'.format(results_file))
    print(train_score, test_score)
    #
    # save_results('MLP', data_params, model_params, train_score, test_score)


def search_model_parameters():
    np.random.seed(np_seed)
    tf.random.set_random_seed(np_seed)

    experiment_path = os.path.join(MODELS_DIR, 'genes_normalization/dae')
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    logging.basicConfig(level=logging.DEBUG, filename=os.path.join(experiment_path, 'log.log'))

    logging.info('Read data')
    data_params = dict(
        features_count=1000,
        rows_count=None,
        filtered_column='Tissue',
        using_values='Whole blood',
        target_column='Age',
        normalize=True,
        use_generator=False,
        noising_method='shift',
        batch_size=128,
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

    input_data, best_genes = load_data(data_params['features_count'])
    input_data = filter_data(input_data, data_params['filtered_column'], data_params['using_values'])

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
    # dropout_rate = [0.25, 0.5, 0.75]
    regularization_param = [10 ** -i for i in range(3, 7)]
    epochs_count = 2000,
    loss = 'mae',
    optimizer = ['adam', 'rmsprop'] #, 'eve's
    learning_rate = 1e-3,

    model_parameters_space = dict(
        layers=layers,
        # drop_rate=dropout_rate,
        activation=activation,
        regularizer_param=regularization_param,
        epochs_count=epochs_count,
        loss=loss,
        optimizer_name=optimizer,
        learning_rate=learning_rate,
        # batch_size=data_params['batch_size'],
    )

    learning_params = dict(
        use_early_stopping=True,
    )

    search_parameters_generator(
        lambda **kwargs: DenoisingAutoencoder(data_params['features_count'], **kwargs),
        'GSE33828',
        input_data,
        best_genes,
        data_params,
        using_metrics=['r2', 'mae'],
        model_parameters_space=model_parameters_space,
        learning_parameters=learning_params,
        cross_validation_parameters=dict(n_splits=5, random_state=sklearn_seed, shuffle=True),
        experiment_dir=experiment_path,
        results_file='cv_results.json',
        search_method_name='random',
        random_n_iter=200,
    )

    # save_search_results(*gs_results, 'MLP_GS')


if __name__ == '__main__':
    # main()
    search_model_parameters()
