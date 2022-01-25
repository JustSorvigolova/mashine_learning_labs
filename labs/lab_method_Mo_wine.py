# Импортируем нужные библиотеки
from sklearn.datasets import load_wine
import matplotlib.pyplot as mpt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import mixture as mx
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import seaborn as sns

# загружаем данные
wine = load_wine()
X_wine = wine.data
Y_wine = wine.target
# -------------------------

# Обучение с учителем, Гаусов наивный Байесов классификатор
xTrain, xTest, yTrain, yTest = train_test_split(X_wine, Y_wine, random_state=0)
# создаем обьект
model = GaussianNB()
# обучаем модель с помощью fit()
model.fit(xTrain, yTrain)
# предсказываем
y_model = model.predict(xTest)
mat = confusion_matrix(yTest, y_model)
# выводим график
sns.heatmap(mat, square=True, annot=True, cbar=False)
mpt.xlabel('predicted values')
mpt.ylabel('true values')
mpt.show()
# -------------------------

# Без учителя. Кластеризация
# создаем модель
model0 = mx.GaussianMixture(n_components=18, covariance_type='full')
# обучаем модель с помощью fit()
model0.fit(X_wine)
# предсказываем
labels = model0.predict(X_wine)
# выводим график
mpt.scatter(X_wine[:, 0], X_wine[:, -1], c=labels, alpha=0.5)
mpt.show()

# -------------------------
# Линейная регрессия
# создаем модель
model = LogisticRegression(multi_class='multinomial')
#  обучаем модель с помощью fit()
model.fit(xTrain, yTrain)
# предсказываем
leaner_reg_predict = model.predict(xTest)
# выводим график
mat = confusion_matrix(yTest, leaner_reg_predict)
sns.heatmap(mat, square=True, annot=True, cbar=False)
mpt.xlabel('predicted value')
mpt.ylabel('true value')
mpt.show()

# -------------------------
# Понижение размерности
# создаем модель
model = PCA(n_components=3)
# обучаем модель с помощью fit()
model.fit(xTrain)
# транформируем в 2-х мерное и выводим результат
X_2D = model.transform(xTrain)
results = pd.DataFrame()
results['PCA1'] = X_2D[:, 0]
results['PCA2'] = X_2D[:, 1]
results['hue'] = yTrain
sns.lmplot(x='PCA1', y='PCA2', hue="hue", data=results, fit_reg=False)
mpt.show()
