from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam, Adamax, Nadam
from Eve.Eve import Eve


__name_to_optimizer = {
    'rmsprop': RMSprop,
    'adam': Adam,
    'sgd': SGD,
    'adagrad': Adagrad,
    'adadelta': Adadelta,
    'adamax': Adamax,
    'nadam': Nadam,
    'eve': Eve,
}


def make_optimizer(optimizer_name='adam', **optimizer_params):
    try:
        optimizer = __name_to_optimizer[optimizer_name]
    except KeyError as e:
        raise ValueError('Undefined optimizer: {}'.format(e.args[0]))

    return optimizer(**optimizer_params)
