import pandas as pd
import numpy as np

df = pd.read_excel("Price Predict.xlsx")
df.head()

x = df["Weight"]
y = df["Price"]

mean_x = np.mean(x)
mean_y = np.mean(y)

mean_x
mean_y

dev_x = x-mean_x
dev_y = y-mean_y

dev_x
dev_y

m = np.sum(dev_x * dev_y)/np.sum(dev_x**2)
m

c = mean_y - (m * mean_x)
c

price_predict = (m*6.5)+c
price_predict


#Sklearn
from sklearn.linear_model import LinearRegression 
reg = LinearRegression() 
reg.fit(df[["Weight"]],df[["Price"]])

reg.coef_
reg.intercept_

df2 = df.copy()
df2

df2["Predicted_price"] = reg.predict(df[["Weight"]]) 
df2

df2["residuals"] = df2["Price"] - df2["Predicted_price"]
df2

reg.score(df[["Weight"]],df[["Price"]])

from sklearn.metrics import mean_squared_error, mean_absolute_error

mse = mean_squared_error(df2[["Predicted_price"]],df2[["Price"]])
mse

mae = mean_absolute_error(df2[["Predicted_price"]],df2[["Price"]])
mae

