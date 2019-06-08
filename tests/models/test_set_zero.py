import pytest
from module.models.dae import set_zero
import numpy as np
# from sklearn.model_selection import KFold


def test_set_zero():
    x = np.random.rand(2, 20)
    v = 0.2
    print(x)
    corrupt_x = set_zero(x, v)

    for input_line, corrupt_line in zip(x, corrupt_x):
        assert (sum(corrupt_line == 0.) > sum(input_line == 0.))


from definitions import *


def test_k_fold():
    kfold = KFold(shuffle=True, random_state=np_seed, n_splits=3)
    a = np.random.rand(100, 1)

    indexes = kfold.split(a)
    indexes2 = kfold.split(a)

    print([i for i in indexes])
    print([i for i in indexes2])