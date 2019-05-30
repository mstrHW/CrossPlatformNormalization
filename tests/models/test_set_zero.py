import pytest
from module.models.dae import set_zero
import numpy as np


def test_set_zero():
    x = np.random.rand(2, 20)
    v = 0.2
    print(x)
    corrupt_x = set_zero(x, v)

    for input_line, corrupt_line in zip(x, corrupt_x):
        assert (sum(corrupt_line == 0.) > sum(input_line == 0.))
