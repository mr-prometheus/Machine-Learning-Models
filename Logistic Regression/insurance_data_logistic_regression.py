import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#%matplotlib inline

df = pd.read_csv('insurance_data.csv')
print(df.head())
plt.scatter(df.age,df.bought_insurance,marker = '+',color = 'red')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(df[['age']],df.bought_insurance,train_size=0.8)
print(X_test)
model = LogisticRegression()
model.fit(X_train, y_train)
y_predicted = model.predict(X_test)
model.predict_proba(X_test)
print(model.score(X_test,y_test))