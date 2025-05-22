# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the standard libraries.

2. Upload the dataset and check for any null values using .isnull() function.

3. Import LabelEncoder and encode the dataset.

4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

5. Predict the values of arrays.

6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

7. Predict the values of array.

8. Apply to new unknown values.

## Program:
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

Developed by:VINOTHINI T

RegisterNumber: 212223040245
```python
import pandas as pd


data = pd.read_csv("Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()

x = data[["Position", "Level"]]
y = data["Salary"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
y_pred = df.predict(x_test)

from sklearn import metrics
mse = metrics.mean_squared_error(y_test, y_pred)
mse

r2 = metrics.r2_score(y_test, y_pred)
r2

dt.predict([[5,6]])
```

## Output:

![Image](https://github.com/user-attachments/assets/5881afbf-10e5-4af9-9d63-946b04849626)
![Image](https://github.com/user-attachments/assets/a3431017-2556-4e1b-b477-dd6d3522cae3)
![Image](https://github.com/user-attachments/assets/920fbb7a-db15-48e4-8e72-25bad941dcc5)
![Image](https://github.com/user-attachments/assets/1f98dc58-f24e-45a6-a2bc-649dfd5c83b8)
![Image](https://github.com/user-attachments/assets/7e6991f8-f8b2-48bb-b25c-6f91ca23fa23)
![Image](https://github.com/user-attachments/assets/89743852-da51-4448-a583-e9c4b9ef4fb2)
![Image](https://github.com/user-attachments/assets/c1866240-8656-4236-8fe5-0e61abe1baec)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
