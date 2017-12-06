import numpy as np
import krig_book as kg
import pytest as pt

# TO RUN TESTS
# python -m pytest test_kriging_book.py


def test_distance():
    a = np.array([1, 0])
    b = np.array([0, 0])
    d = kg.distance(a, b)
    assert d == 1


def test_gamma_gaussian():
    d = np.array([2])
    c = kg.gamma_gaussian(0, 1478, 2.68, d)
    assert c == pt.approx(631.14609587)
