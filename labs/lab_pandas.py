import numpy as np
import pandas as pd
# Имеются данные о среднем бале каждого студента факультет за семестр
index = [('Rustanov', 'Akbar', 100), ('Abdykasymov', 'Erlan', 100),
         ('Nurushev', 'Talgat', 100),
         ('Hrusheva', 'davletgul', 100),
         ('Suiymbaev ', 'Daniyar', 60),
         ('Kutuzova', 'Alena', 60)]
grop_and_sex = [('it-119', 'm'), ('it-119', 'm'), ('it-119', 'm'), ('it-119', 'f'), ('it-119', 'm'), ('it-119', 'f')]
fakultet = pd.Series(grop_and_sex, index=index)
#
index = pd.MultiIndex.from_tuples(index)
# # создаем обьект series индексирование фамилие и имя и добавить мультииндиксирование  номер группы и пол
print(index)

print(fakultet)
#
fakultet = fakultet.reindex(index)
print('-------------')
# # произвести выборку по успеваемости найти отличников и  неумпевающих по полу
fall = fakultet[:,:,60]
print('неуспевающие: ', fall)
cool = fakultet[:,:,100]
print('отличники : ', cool)

falcultets = {('Rustanov Akbar', 100),
              ('Abdykasymov Erlan', 100),
              ('Nurushev Talgat', 100),
              ('Hrusheva', 'davletgul', 100),
              ('Suiymbaev ', 'Daniyar', 60),
              ('Kutuzova', 'Alena', 60)}
print('-------------')
name = ['Rustanov Akbar', 'Abdykasymov Erlan','Nurushev Talgat', 'Hrusheva davletgul', 'Suiymbaev Daniyar', 'Kutuzova Alena']
sex = ['m', 'm', 'm', 'f', 'm', 'f']
group = ['it-119', 'ir-119','it-119', 'ir-119','it-119', 'ir-119']
data = [100, 65, 68, 89, 63,68]
df = pd.DataFrame({"name": name, "sex": sex, "group": group, "gpa": data})

print(df, '\n')
# найти самый высокий бал
print(df.max())




