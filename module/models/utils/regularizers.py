from keras.regularizers import l1, l2, l1_l2


__name_to_keras_regularizer = {
    'l1': l1,
    'l2': l2,
    'l1_l2': lambda parameter: l1_l2(l1=parameter, l2=parameter),
}


def make_regularizer(regularizer_name, regularizer_param):
    try:
        regularizer = __name_to_keras_regularizer[regularizer_name]
    except KeyError as e:
        raise ValueError('Undefined regularizer: {}'.format(e.args[0]))

    return regularizer(regularizer_param)
