import numpy as np
import matplotlib.pyplot as plt
# import pdb as pdb
import krig_book as kg
import numpy.linalg as ln

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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

X = np.stack((x1, x2), axis=-1)

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
h, d = kg.periodgram(X, y)

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

c_xi_xj = kg.build_C_xi_xj(X)
c_xi_xj = kg.add_unitary_column(c_xi_xj)
print(c_xi_xj)


x_in = np.array([13, 19])
c_xinput_xi_xi = kg.build_C_x_xinput(X, x_in)
c_xinput_xi_xi = kg.add_unitary_column(c_xinput_xi_xi)
print(c_xinput_xi_xi)


lam = np.dot(ln.inv(c_xi_xj), c_xinput_xi_xi)
y_xinput = np.dot(lam[:(lam.shape[0]-1)].reshape(1, lam.shape[0]-1), y)
print(y_xinput)

x1_in = np.arange(13, 16.5, 0.5)
x2_in = np.arange(17, 19.5, 0.5)
A, B = np.meshgrid(x1_in, x2_in)


XX_in = np.meshgrid(x1_in, x2_in)
XX_1 = XX_in[0].reshape(35, 1)
XX_2 = XX_in[1].reshape(35, 1)

XX_in = np.stack((XX_1, XX_2), axis=-1)

XX_in = np.array(XX_in).reshape(35, 2)

print(XX_in)

#XX_in = np.array([[13, 17]])
y_out = kg.kriging_interp(X, y, XX_in)

print(y_out)



fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

CS = ax2.contourf(A, B, y_out.reshape(5, 7), 20, cmap=cm.inferno)

CS2 = ax2.contour(CS, levels=CS.levels,
                  colors='k', alpha=0.5)
#circle = ax2.scatter(X[:,0], X[:,1], s=20, c='blue', alpha=0.75)
fig2.colorbar(CS, shrink=0.5, aspect=5)

ax2.set_xlabel(r'$x_1$')
ax2.set_ylabel(r'$x_2$')

plt.savefig('kriging.pdf')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(A, B, y_out.reshape(5, 7), cmap=cm.inferno,
                       linewidth=0, antialiased=True)

ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')

ax.set_zlabel(r'$Kriging$')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.savefig('kriging_3D.pdf')
plt.show()