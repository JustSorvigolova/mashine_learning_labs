# Импортируем все нам нужные библиотеки
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as mpt

sns.set()
fb = pd.read_csv('FremontBridge.csv', index_col='Date', parse_dates=True)
print(fb)
fb.columns = ['East', 'West']
fb['Total'] = fb['East'] + fb['West']
fb['Total'] = fb.eval('East + West')


gby_time = fb.groupby(fb.index.time).mean()
hourly_ticks = 60 * 60 * np.arange(24)
print('Hourly Ticks :', hourly_ticks)
gby_time.plot(xticks=hourly_ticks, figsize=(12, 6))
plt.xticks(rotation='vertical')
plt.show()


xTrain, xTest, yTrain, yTest = train_test_split(gby_time, hourly_ticks, random_state=0)
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

# # группируем по недели и дня
weekday_mask = np.where(fb.index.weekday < 5, 'weekday', 'weekend')
gby_hourly = fb.groupby([weekday_mask, fb.index.time]).mean()
print(gby_hourly)
hourly_ticks = 60 * 60 * np.arange(24)
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
gby_hourly.loc['будний день', :].plot(ax=ax[0], xticks=hourly_ticks, title='Будни', style=[':', '--', '-.'])
gby_hourly.loc['выходной', :].plot(ax=ax[1], xticks=hourly_ticks, title='Выходные', style=[':', '--', '-.'])




