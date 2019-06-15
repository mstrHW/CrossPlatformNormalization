from keras.layers import LeakyReLU, PReLU, ELU


__name_to_activation = {
    'lrelu': LeakyReLU,
    'prelu': PReLU,
    'elu': ELU,
}


def make_activation(activation_name):
    try:
        activation = __name_to_activation[activation_name]
    except KeyError as e:
        raise ValueError('Undefined optimizer: {}'.format(e.args[0]))

    return activation
