# пример обучения без учителя, понижение размероности набора данных Iris
import seaborn as sns
import matplotlib
# 1
from sklearn.decomposition import PCA

iris = sns.load_dataset('iris')

sns.pairplot(iris, hue='species')

# 2 создание экземпляра модели
model = PCA(n_components=2)

# 3
# извлечение матрицы признаков
X_iris = iris.drop('species', axis=1)

# извлечение целевого массива
Y_iris = iris['species']

# 4 обучение модели на данных
model.fit(X_iris)

# преобразование данных в двумерные
X_2D = model.transform(X_iris)

iris['PCA1'] = X_2D[:, 0]

iris['PCA2'] = X_2D[:, 1]

sns.lmplot(x='PCA1', y='PCA2', hue='species', data=iris, fit_reg=False)
#
matplotlib.pyplot.show()

'''
1. выбор модели из библиотеки Scikit-Learn
2. создание модели и выбор гиперпараметров
3. компановка данных в матрицу признаков и целевой вектор
4. обучение модели на своих данных представленных подсредством метода fit() экземпляра
    модели
5. применение модели к новым данным:
    - в случае МО с учителем метки для неизвестных данных обычно предсказываются с
      помощью метода predict()
    - в случае МО без учителя выполняется преоброзование свойств данных или вывод их
      значений подсредством методов transform() или predict()

'''
