from keras.layers import Input, Dense
from keras.models import Model
from module.models.dae import DenoisingAutoencoder
from module.models.utils.metrics import make_sklearn_metric, make_metric
from module.models.utils.optimizers import make_optimizer


class DAEwithPredictor(DenoisingAutoencoder):
    def __init__(self, features_count, **kwargs):
        DenoisingAutoencoder.__init__(self, features_count, **kwargs)
        self.predictor = None

    def build_model(self):
        input_layer = Input(shape=(self.features_count,))
        encoded = self.build_encoder(input_layer, self.layers[0], self.activation)
        decoded = self.build_decoder(encoded, self.layers[1], self.activation, self.output_activation)
        predicted = Dense(1, activation=self.output_activation, name='predictor')(encoded)

        model = Model(input_layer, [decoded, predicted])

        opt = make_optimizer(self.optimizer_name, lr=self.learning_rate)
        model.compile(loss=self.loss, optimizer=opt, metrics=[make_metric('r2')])

        return model

    def score(self, test_data, metrics, scaler=None):
        if metrics is str:
            metrics = [metrics]

        y_preds = self.model.predict(test_data[0])
        decoder_pred, predictor_pred = y_preds

        ys = test_data[1]
        decoder_y, predictor_y = ys

        if scaler is not None:
            decoder_pred = scaler.inverse_transform(decoder_pred)
            decoder_y = scaler.inverse_transform(decoder_y)

        scores = dict()
        for metric_name in metrics:
            scores['decoder_{}'.format(metric_name)] = make_sklearn_metric(metric_name)(decoder_y, decoder_pred)

        for metric_name in metrics:
            scores['predictor_{}'.format(metric_name)] = make_sklearn_metric(metric_name)(predictor_y, predictor_pred)

        return scores
