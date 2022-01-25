import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.ensemble import RandomForestClassifier

df1 = pd.read_csv("Heart_Disease_Prediction.csv")

print(df1.head(5))
print(df1.shape)
print(df1.info())
label_1 = df1["Heart Disease"]
df1.drop("Heart Disease", axis=1, inplace=True)
print(label_1.value_counts())
label_1.value_counts().plot(kind="bar")

categorical_features = ["Sex", "Chest pain type", "FBS over 120", "EKG results",
                        "Exercise angina", "Slope of ST", "Number of vessels fluro", "Thallium"]

df1[categorical_features] = df1[categorical_features].astype("category")
print('------------------')
print(df1.info())

continuous_features = set(df1.columns) - set(categorical_features)
scaler = MinMaxScaler()
df1_norm = df1.copy()
df1_norm[list(continuous_features)] = scaler.fit_transform(df1[list(continuous_features)])
scaler.fit_transform(df1[list(continuous_features)])
X_new = SelectKBest(k=5, score_func=chi2).fit_transform(df1_norm, label_1)
print("Отбор методом случайных х -квадрат", X_new)
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=5)
X_new = rfe.fit_transform(df1_norm, label_1)

print("Отбор методом случайных деревьев", X_new)
clf = RandomForestClassifier()
clf.fit(df1_norm, label_1)
# Создаем диаграмму
plt.figure(figsize=(12, 12))
plt.bar(df1_norm.columns, clf.feature_importances_)
plt.xticks(rotation=45)
plt.show()
