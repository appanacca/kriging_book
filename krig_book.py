import numpy as np
# import pdb as pdb


def distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def periodgram(x, y):
    h = []
    d = []
    for i in np.arange(len(x[:, 0])):
        for j in np.arange(len(x[:, 0])):
            d.append(distance(x[i], x[j]))
            h.append(0.5*(y[i] - y[j])**2)
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


def build_C_xi_xj(X):
    """
    xi = numpy array of length N that contains the FIRST data point feature
    xj = numpy array of length N that contains the SECOND data point feature
    build the NxN matrix that represent the correlation between
    each data points
    """
    c = np.zeros((len(X[:, 0]), len(X[:, 0])))
    # it can be imporved:
    # 1) there is a better way to iterate in python
    # 2) the c matrix is symmetric so we need to iterate only
    #    the upper triangular part
    for i in np.arange(len(X[:, 0])):
        for j in np.arange(len(X[:, 0])):
            d = np.array([distance(X[i], X[j])])
            c[i, j] = gamma_gaussian(0, 1478, 2.68, d)
    return c


def build_C_x_xinput(x, x_input):
    """
    xi = numpy array of length N that contains the FIRST data point feature
    xj = numpy array of length N that contains the SECOND data point feature
    build the NxN matrix that represent the correlation between
    each data points
    """
    assert len(x[0, :]) == len(x_input)
    c = np.zeros((len(x[:, 0]), 1))

    # it can be imporved:
    # 1) there is a better way to iterate in python
    # 2) the c matrix is symmetric so we need to iterate only
    #    the upper triangular part
    for i in np.arange(len(x[:, 0])):
            d = np.array([distance(x[i], x_input)])
            c[i] = gamma_gaussian(0, 1478, 2.68, d)
    return c
