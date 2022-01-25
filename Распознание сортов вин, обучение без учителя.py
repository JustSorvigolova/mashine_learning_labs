from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# импорт класса линейной регрессии
from sklearn.linear_model import LinearRegression

import seaborn as sns

wine = load_wine()

# получение массива признаков и целевой массив
X = wine.data

print(X.shape)

y = wine.target

print(y.shape)

# понижаем размерность до 2
iso = Isomap(n_components=2)
iso.fit(wine.data)
data_projected = iso.transform(wine.data)

print(data_projected.shape)
# визуализируем пониженное (двухмерное) представление
'''
plt.scatter(data_projected[:, 0], data_projected[:, 1], c = wine.target, edgecolor = 'none', alpha =0.5, cmap = plt.cm.get_cmap('Spectral', 10))
plt.colorbar(label = 'wine label', ticks = range(10))
plt.clim(-0.5, 9.5)
plt.show()
'''
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)
model = GaussianNB()
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)

print(accuracy_score(ytest, y_model))

# построение матрицы различий

mat = confusion_matrix(ytest, y_model)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')

# plt.show()
