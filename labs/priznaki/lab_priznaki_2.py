import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.ensemble import RandomForestClassifier

df2 = pd.read_csv("affairs.csv")
print(df2.info())
label_1 = df2["religious"]
df2.drop("religious", axis=1, inplace=True)
print(label_1.value_counts())
label_1.value_counts().plot(kind="bar")
categorical_features = ["Unnamed: 0", "rate_marriage", "age", "yrs_married",
                        "children", "educ", "occupation", "occupation_husb"]

df2[categorical_features] = df2[categorical_features].astype("category")
continuous_features = set(df2.columns) - set(categorical_features)
scaler = MinMaxScaler()
df2_norm = df2.copy()
df2_norm[list(continuous_features)] = scaler.fit_transform(df2[list(continuous_features)])
scaler.fit_transform(df2[list(continuous_features)])
X_new = SelectKBest(k=5, score_func=chi2).fit_transform(df2_norm, label_1)
print("Отбор методом случайных х -квадрат", X_new)
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=5)
X_new = rfe.fit_transform(df2_norm, label_1)

print("Отбор методом случайных деревьев", X_new)
clf = RandomForestClassifier()
clf.fit(df2_norm, label_1)
# Создаем диаграмму
plt.figure(figsize=(12, 12))
plt.bar(df2_norm.columns, clf.feature_importances_)
plt.xticks(rotation=45)
plt.show()
