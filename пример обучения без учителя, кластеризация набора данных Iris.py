# пример обучения без учителя, кластеризация набора данных Iris
import seaborn as sns
import matplotlib
from sklearn import mixture

iris = sns.load_dataset('iris')

# sns.pairplot(iris, hue = 'species')
# matplotlib.pyplot.show()

# 3
# извлечение матрицы признаков
X_iris = iris.drop('species', axis=1)
print(X_iris)
# извлечение целевого массива
Y_iris = iris['species']

# 2 создание экземпляра модели

model = mixture.GaussianMixture(n_components=3, covariance_type='full')

# 4 обучение модели на данных
model.fit(X_iris)

# определяем метки кластеров
y_gmm = model.predict(X_iris)

iris['cluster'] = y_gmm
# ***********************************************
# 2 создание экземпляра модели
from sklearn.decomposition import PCA

model2 = PCA(n_components=2)

# 4 обучение модели на данных
model2.fit(X_iris)

# преоброзование данных в двумерные
X_2D = model2.transform(X_iris)

iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]
# ***********************************************

# sns.lmplot(x='PCA1', y='PCA2', data=iris, hue='species', col='cluster', fit_reg=False)
#
# matplotlib.pyplot.show()
