import numpy as np
import matplotlib.pyplot as plt
# import pdb as pdb

x1 = np.array([14.04,
               14.33,
               15.39,
               13.76,
               14.59,
               13.48,
               15.86,
               15.61,
               13.29,
               14.81])

x2 = np.array([18.76,
               18.54,
               17.05,
               17.54,
               17.84,
               17.21,
               17.61,
               18.85,
               18.20,
               18.15])

y = np.array([77.88,
              71.03,
              27.97,
              63.41,
              57.76,
              64.63,
              33.54,
              65.64,
              92.53,
              67.06])


def fun(a, b):
    idx = np.where(((x1 == a) & (x2 == b)))
    return y[idx]


def periodgram(x1, x2):
    h = []
    d = []
    for i in np.arange(len(x1)):
        for j in np.arange(len(x2)):
            # pdb.set_trace()
            d.append(np.sqrt((x1[i] - x1[j])**2 + (x2[i] - x2[j])**2))
            h.append(0.5*(fun(x1[i], x2[i]) - fun(x1[j], x2[j]))**2)
    return h, d


def avg_periodgram(h, d, lag):
    d_avg = []
    h_avg = []
    h = np.array(h)
    d = np.array(d)
    steps = np.arange(min(d)+lag, max(d)+lag, lag)
    for i in steps:
        ist = h[(d < i) & (d > (i - lag))]
        d_ist = d[(d < i) & (d > (i - lag))]
        n = len(ist)
        h_avg.append(np.sum(ist)/n)
        d_avg.append(np.sum(d_ist)/n)
    return h_avg, d_avg


def gamma_gaussian(C0, C1, R, h):
    gamma = np.zeros(len(h))
    for i in np.arange(len(h)):
        if h[i] > 0:
            gamma[i] = (C0 + C1*(1 - np.exp((-h[i]**2)/(R**2))))
        else:
            gamma[i] = 0
    return gamma


def build_C_xi_xj(xi, xj):
    """
    xi = numpy array of length N that contains the FIRST data point feature
    xj = numpy array of length N that contains the SECOND data point feature
    build the NxN matrix that represent the correlation between
    each data points
    """
    assert len(xi) == len(xj)
    c = np.zeros((len(xi), len(xi)))

    # it can be imporved:
    # 1) there is a better way to iterate in python
    # 2) the c matrix is symmetric so we need to iterate only
    #    the upper triangular part
    for i in np.arange(len(xj)):
        for j in np.arange(len(xj)):
            d = np.array([np.sqrt((xi[i] - xi[j])**2 + (xi[i] - xi[j])**2)])
            c[i, j] = gamma_gaussian(0, 1478, 2.68, d)
    return c

def build_C_x_xinput(x, x_input):
    """
    xi = numpy array of length N that contains the FIRST data point feature
    xj = numpy array of length N that contains the SECOND data point feature
    build the NxN matrix that represent the correlation between
    each data points
    """
    assert len(x) == len(x_input)
    c = np.zeros((len(x), len(x_input)))

    # it can be imporved:
    # 1) there is a better way to iterate in python
    # 2) the c matrix is symmetric so we need to iterate only
    #    the upper triangular part
    # 3) i need to define a distance function
    for i in np.arange(len(x)):
        for j in np.arange(len(x_input)):
            d = np.array([np.sqrt((x[i] - x_input[j])**2 + (x[i] - x_input[j])**2)])
            c[i, j] = gamma_gaussian(0, 1478, 2.68, d)
    return c

