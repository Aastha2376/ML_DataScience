import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


df = pd.read_csv(r"C:\Users\Aastha Arora\Downloads\ML+Resources\homeprices.csv")


print(df)


plt.xlabel('Area')
plt.ylabel('Price')
plt.scatter(df['Area'], df['Price'], color='red', marker='+')


plt.show()


reg = LinearRegression()


reg.fit(df[['Area']], df['Price'])


predicted_price = reg.predict([[3300]])
print(f"Predicted price for a 3300 sq ft area: {predicted_price[0]}")


print(f"Coefficient (Slope) of the model: {reg.coef_[0]}")


print(f"Intercept of the model: {reg.intercept_}")
