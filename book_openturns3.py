import numpy as np
import matplotlib.pyplot as plt
# import pdb as pdb
import openturns as ot

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

font = {'family': 'normal',
        'weight': 'normal',
        'size': 22,
        }

rc('font', **font)


font1 = {'family': 'serif',
         'color':  'black',
         'weight': 'normal',
         'size': 26,
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


z = np.array([77.88,
              71.03,
              27.97,
              63.41,
              57.76,
              64.63,
              33.54,
              65.64,
              92.53,
              67.06])


x = np.column_stack((x1, x2))
X = ot.Sample(x)
z = ot.Sample(np.reshape(z, (len(z), 1)))


dimension = len(x[0])

basis = ot.ConstantBasisFactory(dimension).build()
# basis = ot.LinearBasisFactory(dimension).build()
# basis = ot.QuadraticBasisFactory(dimension).build()

covarianceModel = ot.SquaredExponential([10], [1.895])
# covarianceModel = ot.MaternModel()

algo = ot.KrigingAlgorithm(X, z, covarianceModel, basis)
# algo.setNoise([0.2]*len(z)) # nugget
algo.run()


result = algo.getResult()
metamodel = result.getMetaModel()


a = np.linspace(13, 16, 100)
b = np.linspace(17, 19, 100)
A, B = np.meshgrid(a, b)
aa = np.reshape(A, (10000, 1))
bb = np.reshape(B, (10000, 1))
c = np.column_stack((aa, bb))
C = np.array(metamodel(c))
cc = np.reshape(C, (100, 100))

fig = plt.figure(figsize=(16, 6), dpi=250, facecolor='w', edgecolor='k')
ax1 = fig.add_subplot(1, 2, 2)
CS = ax1.contourf(A, B, cc, 20, cmap=cm.inferno)  

CS2 = ax1.contour(CS, levels=CS.levels,
                  colors='k', alpha=0.5)

circle = ax1.scatter(x1, x2, s=40, c='blue', alpha=0.75)
ax1.set_xlabel(r'$x_1$', fontdict=font1)
ax1.set_ylabel(r'$x_2$', fontdict=font1)
fig.colorbar(CS, shrink=0.5, aspect=5)

ax2 = fig.add_subplot(1, 2, 1, projection='3d')

surf = ax2.plot_surface(A, B, cc, cmap=cm.inferno,
                       linewidth=0, antialiased=True, alpha=0.8)
ax2.scatter(X[:, 0], X[:, 1], z, s=40, c='blue', alpha=1)

# ax2.set_xlabel(r'$x_1$', fontdict=font1)
# ax2.set_ylabel(r'$x_2$', fontdict=font1)
# ax2.set_zlabel(r'$\hat{y}$', fontdict=font1)
# fig.colorbar(surf, shrink=0.5, aspect=5)
plt.subplots_adjust(wspace=0.17, hspace=0)

plt.savefig('kriging_openturns3.pdf', bbox_inches='tight', dpi=500)
plt.show()
