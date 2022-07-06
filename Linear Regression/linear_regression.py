import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
df = pd.read_csv("homeprices.csv")
plt.xlabel('area(sqft)')
plt.ylabel("prices")
plt.scatter(df.area,df.price,color = 'red',marker = '+')
reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.price)
print(reg.predict([[3300]]))
plt.show()