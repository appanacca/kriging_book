import numpy as np
import matplotlib.pyplot as plt
# import pdb as pdb
import krig_book as kg

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

fig, ax = plt.subplots(dpi=250)
plt.scatter(x1, x2, c=y)
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
plt.colorbar()
plt.show()


h = []
d = []
h, d = kg.periodgram(x1, x2)

fig, ax = plt.subplots(dpi=250)
plt.scatter(d, h)
ax.set_xlabel(r'$h$')
ax.set_ylabel(r'$\gamma(h)$')
plt.show()


h_avg = []
d_avg = []
h_avg, d_avg = kg.avg_periodgram(h, d, 0.25)

dist = np.arange(0, 2.75, 0.01)
gg = kg.gamma_gaussian(0, 1478, 2.68, dist)


fig, ax = plt.subplots(dpi=250)
plt.scatter(d_avg, h_avg)
plt.plot(dist, gg, 'r')
ax.set_xlabel(r'$h$')
ax.set_ylabel(r'$\gamma_{avg}(h)$')
plt.show()

c_xi_xj = np.zeros((len(x1), len(x2)))
c_xi_xj = kg.build_C_xi_xj(x1, x2)


c_xinput_xi = np.zeros((len(x1), len(x2)))
x_in = np.array([15,18])
X = np.array([x1, x2])
print(X)
c_xinput_xi_xi = kg.build_C_x_xinput(X, x_in)