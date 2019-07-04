from keras.layers import Input, Dense
from keras.models import Model
import numpy as np

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

    def score(self, test_generator, metrics):
        if metrics is str:
            metrics = [metrics]

        ae_y_preds = []
        ae_ys = []

        predictor_y_preds = []
        predictor_ys = []

        for batch_X, batch_y in test_generator:
            batch_predicts = self.model.predict_on_batch(batch_X)

            ae_y_preds.append(batch_predicts[0])
            predictor_y_preds.append(batch_predicts[1])

            ae_ys.append(batch_y[0])
            predictor_ys.append(batch_y[1])

        if len(ae_y_preds) > 1:
            ae_y_preds = np.concatenate(ae_y_preds, axis=0)
            predictor_y_preds = np.concatenate(predictor_y_preds, axis=0)
            ae_ys = np.concatenate(ae_ys, axis=0)
            predictor_ys = np.concatenate(predictor_ys, axis=0)
        else:
            ae_y_preds = np.array(ae_y_preds[0])
            predictor_y_preds = np.array(predictor_y_preds[0])
            ae_ys = np.array(ae_ys[0])
            predictor_ys = np.array(predictor_ys[0])

        scores = dict()
        for metric_name in metrics:
            scores['ae_{}'.format(metric_name)] = make_sklearn_metric(metric_name)(ae_ys, ae_y_preds)

        scores = dict()
        for metric_name in metrics:
            scores['predictor_{}'.format(metric_name)] = make_sklearn_metric(metric_name)(predictor_ys, predictor_y_preds)

        return scores
