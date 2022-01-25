import seaborn as sns
import matplotlib
from sklearn import mixture
from sklearn.datasets import load_wine
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
wine = load_wine()
# X_wine = wine.data
#
# Y_wine = wine.target
# sns.pairplot(X_wine, hue = 'species')
# matplotlib.pyplot.show()
# model = mixture.GaussianMixture(n_components=2, covariance_type='full')

X1 = load_wine()["data"][:, [1, 2]]  # two clusters
print(X1)
# model.fit(X_wine)
# y_gmm = model.predict(X_wine)
# wine['cluster'] = y_gmm
# model2 = PCA(n_components=2)
# model2.fit(X_wine)
# X_2D = model2.transform(X_wine)
# wine['PCA1'] = X_2D[:, 0]
# wine['PCA2'] = X_2D[:, 1]
# sns.lmplot(x='PCA1', y='PCA2', data=wine, hue='species', col='cluster', fit_reg=False)
#
# matplotlib.pyplot.show()
