import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r"C:\Users\Aastha Arora\Downloads\ML+Resources\50_Startups.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1].values


ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


regressor = LinearRegression()
regressor.fit(X_train, y_train)


y_pred = regressor.predict(X_test)


df_pred = pd.DataFrame({'Real Values': y_test, 'Predicted Values': y_pred})
print(df_pred)


