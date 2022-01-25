import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier

df3 = pd.read_csv("accidental-deaths-in-usa-monthly.csv")

print(df3.head(5))
print(df3.info())
print(df3.shape)

label_1 = df3["Month"]
df3.drop("Month", axis=1, inplace=True)
print(label_1.value_counts())
label_1.value_counts().plot(kind="bar")

categorical_features = ["Accidental deaths in USA: monthly, 1973 ? 1978"]
df3[categorical_features] = df3[categorical_features].astype("category")

continuous_features = set(df3.columns) - set(categorical_features)
scaler = MinMaxScaler()
df3_norm = df3.copy()
df3_norm[list(continuous_features)] = scaler.fit_transform(df3[list(continuous_features)])
print("Отбор методом случайных х -квадрат")
X_new = SelectKBest(k=5, score_func=chi2).fit_transform(df3_norm, label_1)
print(X_new)
print("Отбор методом случайных деревьев")
clf = RandomForestClassifier()
clf.fit(df3_norm, label_1)
# Создаем диаграмму
plt.figure(figsize=(12, 12))
plt.bar(df3_norm.columns, clf.feature_importances_)
plt.xticks(rotation=45)
plt.show()
