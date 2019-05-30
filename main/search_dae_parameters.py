import logging
from module.models.dae_keras import DenoisingAutoencoder
from module.data_processing import data_processing
from module.models.utils import save_results

logging.basicConfig(level=logging.DEBUG)


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

    train_data, test_data = data_processing.main(True, **data_params)

    logging.debug('Create model')

    model_params = dict(
        layers=[
            (800, 600, 400, 200),
            (200, 400, 600, 800),
        ],
        activation='elu',
        output_activation='linear',
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

    logging.debug('Fit model')
    model.fit_generator(train_data, test_data, **learning_params)

    train_score = model.score_generator(train_data)
    test_score = model.score_generator(test_data)

    save_results('DAE', data_params, model_params, learning_params, train_score, test_score)


if __name__ == '__main__':
    main()
