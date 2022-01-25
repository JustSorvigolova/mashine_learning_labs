import seaborn as sns
from sklearn.model_selection import train_test_split
# подключение гауссова нивного байесовского классификатора
from sklearn.naive_bayes import GaussianNB  # 1

iris = sns.load_dataset('iris')

# извлечение матрицы признаков
X_iris = iris.drop('species', axis=1)
print(X_iris)
# извлечение целевого массива
Y_iris = iris['species']

# создание тренировочного и контрольного набора данных
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, Y_iris, random_state=1)

# 2 создание экземпляра модели
model = GaussianNB()

# 4 обучение модели на данных
model.fit(Xtrain, ytrain)

# 5 предсказываем значения для новых данных
y_model = model.predict(Xtest)

# подключение утилиты для проверки полученных данных
from sklearn.metrics import accuracy_score

# print(accuracy_score(ytest, y_model))
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
