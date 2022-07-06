import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
df = pd.read_csv("canada_per_capita_income.csv")
plt.scatter(df.year, df.price, color = 'red',marker='+')
reg = linear_model.LinearRegression()
