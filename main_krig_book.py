import numpy as np
import matplotlib.pyplot as plt
# import pdb as pdb
import krig_book as kg
import numpy.linalg as ln

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

font = {'family': 'normal',
        'weight': 'normal',
        'size': 20,
        }

rc('font', **font)


font1 = {'family': 'serif',
         'color':  'black',
         'weight': 'normal',
         'size': 24,
         }


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
ax.set_xlabel(r'$x_1$', fontdict=font1)
ax.set_ylabel(r'$x_2$', fontdict=font1)
plt.colorbar()
plt.savefig('DOE_data.pdf', bbox_inches='tight', dpi=500)
plt.show()


h = []
d = []
h, d = kg.periodgram(X, y)


h_avg = []
d_avg = []
h_avg, d_avg = kg.avg_periodgram(h, d, 0.25)

dist = np.arange(0, 2.75, 0.01)
gg = kg.gamma_gaussian(0, 1478, 2.68, dist)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=250, facecolor='w', edgecolor='k')
ax1.scatter(d, h)
ax1.set_xlabel(r'$h$', fontdict=font1)
ax1.set_ylabel(r'$\gamma(h)$', fontdict=font1)
#plt.savefig('semivariogram.pdf', bbox_inches='tight')

ax2.scatter(d_avg, h_avg)
ax2.plot(dist, gg, 'r')
ax2.set_xlabel(r'$h$', fontdict=font1)
ax2.set_ylabel(r'$\gamma_{avg}(h)$', fontdict=font1)
plt.savefig('sem.pdf', bbox_inches='tight')
plt.show()

c_xi_xj = kg.build_C_xi_xj(X)
c_xi_xj = kg.add_unitary_column(c_xi_xj)


x_in = np.array([13, 19])
c_xinput_xi_xi = kg.build_C_x_xinput(X, x_in)
c_xinput_xi_xi = kg.add_unitary_column(c_xinput_xi_xi)


lam = np.dot(ln.inv(c_xi_xj), c_xinput_xi_xi)
y_xinput = np.dot(lam[:(lam.shape[0]-1)].reshape(1, lam.shape[0]-1), y)

x1_in = np.arange(13, 16.5, 0.05)
x2_in = np.arange(17, 19.5, 0.05)
A, B = np.meshgrid(x1_in, x2_in)


XX_in = np.meshgrid(x1_in, x2_in)
n = len(x1_in)
m = len(x2_in)
nm = n*m

XX_1 = XX_in[0].reshape(nm, 1) 
XX_2 = XX_in[1].reshape(nm, 1)

XX_in = np.stack((XX_1, XX_2), axis=-1)

XX_in = np.array(XX_in).reshape(nm, 2)


y_out = kg.kriging_interp(X, y, XX_in)


fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

CS = ax2.contourf(A, B, y_out.reshape(m, n), 20, cmap=cm.inferno)  # clean this hardcoded dimensions

CS2 = ax2.contour(CS, levels=CS.levels,
                  colors='k', alpha=0.5)

circle = ax2.scatter(x1, x2, s=20, c='blue', alpha=0.75)
fig2.colorbar(CS, shrink=0.5, aspect=5)

ax2.set_xlabel(r'$x_1$', fontdict=font1)
ax2.set_ylabel(r'$x_2$', fontdict=font1)

plt.savefig('kriging.pdf', bbox_inches='tight')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(A, B, y_out.reshape(m, n), cmap=cm.inferno,
                       linewidth=0, antialiased=True)
circle = ax.scatter(x1, x2, y, s=20, c='blue', alpha=0.75)
ax.set_xlabel(r'$x_1$', fontdict=font1)
ax.set_ylabel(r'$x_2$', fontdict=font1)

ax.set_zlabel(r'$\hat{y}$', fontdict=font1)
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.savefig('kriging_3D.pdf', bbox_inches='tight')
plt.show()

print(np.mean(y))