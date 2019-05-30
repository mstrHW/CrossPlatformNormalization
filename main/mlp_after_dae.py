from module.data_processing import data_processing
import logging
from module.models.mlp import MLP
from module.models.utils import save_results
from module.models.dae_keras import DenoisingAutoencoder


def main():
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
        layers=[
            (800, 600, 400, 200),
            (200, 400, 600, 800),
        ],
        activation='elu',
        drop_rate=0.3,
        regularizers_param=1e-4,
    )

    model = DenoisingAutoencoder(data_params['features_count'], **model_params)

    learning_params = dict(
        epochs_count=1500,
        batch_size=data_params['batch_size'],
        learning_rate=0.0001,
        loss='mse',
        patience=200,
    )

    logging.debug('Predict model')
    model.load_trained_model()

    (train_X, train_y) = train_data
    (test_X, test_y) = test_data
    train_X_ = model.predict(train_X)
    test_X_ = model.predict(test_X)

    train_data_ = (train_X_, train_y)
    test_data_ = (test_X_, test_y)


    logging.debug('Create model')

    model_params = dict(
        layers=(1500, 1200, 1000, 800, 700, 500, 256, 128),
        activation='elu',
        drop_rate=0.5,
        regularizers_param=1e-3,
    )

    model = MLP(features_count=data_params['features_count'], **model_params)

    learning_params = dict(
        epochs_count=200,
        batch_size=data_params['batch_size'],
        learning_rate=0.0001,
        loss='mae',
        patience=200,
    )

    logging.debug('Fit model')
    model.fit(train_data, test_data, **learning_params)

    train_score = model.calculate_score(*train_data_)
    test_score = model.calculate_score(*test_data_)

    save_results('MLP_after_DAE', data_params, model_params, learning_params, train_score, test_score)


if __name__ == '__main__':
    main()
