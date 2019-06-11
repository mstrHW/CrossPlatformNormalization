from keras.models import Sequential
from keras.regularizers import l1_l2
from module.models.base_model import BaseModel, make_activation
from keras.layers import Dense, Dropout, BatchNormalization
from module.models.optimizers import make_optimizer





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
        model.compile(loss=self.loss, optimizer=opt)

        return model


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