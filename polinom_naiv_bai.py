# полиномиальный наивный байесовский классификатор
# подключение набора текстов
# подключение библиотеки для построения матрицы коэфициентов
# подключение библиотек с векторизатором TF-IDF
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# извлекаем тексты
data = fetch_20newsgroups()
print(data.target_names)

# 1 вариант
categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'sci.med']

# # построение тренировочноо и тестового набора данных
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

# # построение модели
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
#
# # обучение модели
model.fit(train.data, train.target)
# # применение модели
labels = model.predict(test.data)
#
# # построение матрицы коэфициентов сранения результатов обучения модели
mat = confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=train.target_names,
            yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
#
plt.show()


# # функция определяет текст для передаваемой строки
def predict_category(s, trains=train, models=model):
    pred = models.predict([s])
    return trains.target_names[pred[0]]


print(predict_category('sending a payload to the ISS'))
print(predict_category('discssing islam vs atheism'))
print(predict_category('determining the screen resolution'))
