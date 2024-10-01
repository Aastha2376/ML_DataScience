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

# OneHotEncoding for categorical data (column index 3, assuming it's a categorical column)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict test set results
y_pred = regressor.predict(X_test)

# Creating a DataFrame to compare actual and predicted values
df_pred = pd.DataFrame({'Real Values': y_test, 'Predicted Values': y_pred})
print(df_pred)


