import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

births = pd.read_csv('births.csv')
print(births.head())
births['decade'] = 10 * (births['year'] // 10)
births.pivot_table('births', index='decade', columns='gender', aggfunc='sum')
print(births['decade'])

sns.set()  # Используем стили библиотеки Seaborn
births.pivot_table('births', index='year', columns='gender',
                   aggfunc='sum').plot()
plt.ylabel('общее количество новорожденных в течение года')  # общее количество новорожденных в течение года
plt.show()
# Благодаря сводной таблице и методу plot() мы можем сразу же увидеть ежегодный
# тренд новорожденных по полу. В последние 50 с лишним лет мальчиков рождалось
# больше, чем девочек, примерно на 5 %.

# убираем аномальные значения типа 99 июня.
# убираем их с помощью алгоритма сигма-отсечения
quartiles = np.percentile(births['births'], [25, 50, 75])
mu = quartiles[1]
sig = 0.74 * (quartiles[2] - quartiles[0])
plt.show()

# применяем для фильтрации
# строк, в которых количество новорожденных выходит за пределы этих значений:
births = births.query('(births > @mu - 5 * @sig) & (births < @mu + 5 * @sig)')
# Далее мы устанавливаем целочисленный тип столбца для day
births['day'] = births['day'].astype(int)

# создаем индекс для даты, объединив день, месяц и год
births.index = pd.to_datetime(10000 * births.year +
                              100 * births.month +
                              births.day, format='%Y%m%d')
births['dayofweek'] = births.index.dayofweek  # День недели

# С помощью этого можно построить график дней рождения в зависимости от дня
# недели за несколько десятилетий
births.pivot_table('births', index='dayofweek',
                   columns='decade', aggfunc='mean').plot()
plt.gca().set_xticklabels(['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat',
                           'Sun'])
plt.ylabel('среднее количество новорожденных в день')  # среднее количество новорожденных в день
plt.show()

# также можно построить график рождений в зависимости от дня года
births_by_date = births.pivot_table('births',
                                    [births.index.month, births.index.day])
print(births_by_date.head())
# обрабатываем  29 февраля
births_by_date.index = [pd.datetime(2012, month, day)
                        for (month, day) in births_by_date.index]
print(births_by_date.head())


# Строим график результатов
fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax)
plt.show()
