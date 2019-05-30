import numpy as np


def set_zero(x, v):
    x_noise = x.copy()

    n_samples = x.shape[0]
    n_features = x.shape[1]

    n_corrupt = int(n_features * v)

    for i in range(n_samples):
        mask = np.random.randint(0, n_features, n_corrupt)

        for m in mask:
            x_noise[i][m] = 0.

    return x_noise


def add_gaussian_noise(data):
    n = np.random.normal(0, 0.1, np.shape(data))
    return data + n
