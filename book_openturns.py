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


X = np.column_stack((x1, x2))
X = ot.Sample(X)
z = ot.Sample(np.reshape(z, (len(z), 1)))


dimension = 2

basis = ot.ConstantBasisFactory(dimension).build()
# basis = ot.LinearBasisFactory(dimension).build()
# basis = ot.QuadraticBasisFactory(dimension).build()

covarianceModel = ot.SquaredExponential([0.1], [0.8])
# covarianceModel = ot.MaternModel()

algo = ot.KrigingAlgorithm(X, z, covarianceModel, basis)
# algo.setNoise([0.1]*len(z)) # nugget
algo.run()


result = algo.getResult()
metamodel = result.getMetaModel()


a = np.linspace(13, 16.5, 1000)
b = np.linspace(17, 19.5, 1000)
A, B = np.meshgrid(a, b)
aa = np.reshape(A, (1000000, 1))
bb = np.reshape(B, (1000000, 1))
c = np.column_stack((aa, bb))
C = np.array(metamodel(c))
cc = np.reshape(C, (1000, 1000))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.scatter(X[:, 0], X[:, 1], z, s=30, c='blue', alpha=0.75)
surf = ax.plot_surface(A, B, cc, cmap=cm.plasma,
                       linewidth=0, antialiased=True)

ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_zlabel(r'$\hat{y}$')
fig.colorbar(surf, shrink=0.5, aspect=5)


plt.savefig('kriging_openturns.pdf', bbox_inches='tight')
plt.show()
