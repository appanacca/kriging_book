import numpy as np
import krig_book as kg
import pytest as pt

# TO RUN TESTS
# python -m pytest test_kriging_book.py


def test_fun():
    a = 14.04
    b = 18.76
    c = kg.fun(a, b)
    assert c == 77.88


def test_gamma_gaussian():
    d = np.array([2])
    c = kg.gamma_gaussian(0, 1478, 2.68, d)
    assert c == pt.approx(631.14609587)
