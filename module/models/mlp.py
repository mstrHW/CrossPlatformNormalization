from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization

from module.models.base_model import BaseModel
from module.models.utils.activations import make_activation
from module.models.utils.metrics import make_metric, make_sklearn_metric
from module.models.utils.optimizers import make_optimizer


class MLP(BaseModel):
    def __init__(self, features_count, **kwargs):
        BaseModel.__init__(self, features_count, **kwargs)

    def build_model(self):
        model = Sequential()

        dense_layer = Dense(
            self.layers[0],
            input_shape=(self.features_count,),
            kernel_regularizer=self.regularizer,
            kernel_initializer=self.initializer
        )
        model.add(dense_layer)
        model.add(make_activation(self.activation))
        model.add(BatchNormalization())
        model.add(Dropout(self.drop_rate))

        for layer in self.layers[1:-1]:
            dense_layer = Dense(
                layer,
                kernel_regularizer=self.regularizer,
                kernel_initializer=self.initializer
            )
            model.add(dense_layer)
            model.add(make_activation(self.activation))
            model.add(BatchNormalization())
            model.add(Dropout(self.drop_rate))

        model.add(Dense(self.layers[-1], activation=self.output_activation))

        opt = make_optimizer(self.optimizer_name, lr=self.learning_rate)
        model.compile(loss=self.loss, optimizer=opt, metrics=[make_metric('r2')])

        return model

    def score_sklearn(self, test_data, metrics, scaler=None):
        if metrics is str:
            metrics = [metrics]

        y_preds = self.model.predict(test_data[0])
        ys = test_data[1]

        scores = dict()
        for metric_name in metrics:
            scores['predictor_{}'.format(metric_name)] = make_sklearn_metric(metric_name)(ys, y_preds)

        return scores


if __name__ == '__main__':
    model_params = dict(
        layers=(1500, 1200, 1000, 800, 700, 500, 256, 128, 1),
        activation='elu',
        drop_rate=0.5,
        regularizer_param=1e-3,
        epochs_count=200,
        learning_rate=0.0001,
        loss='mae',
        batch_size=128,
        patience=200,
    )
    MLP(1000, **model_params)
