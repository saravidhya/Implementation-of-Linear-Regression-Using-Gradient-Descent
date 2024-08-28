# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:

To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the standard libraries in python required for finding Gradient Design.
2. Read the dataset file and check any null value using .isnull() method. 
3. Declare the default variables with respective values for linear regression.
4. Calculate the loss using Mean Square Error.
5. Predict the value of y.
6. Plot the graph respect to hours and scores using .scatterplot() method for Linear Regression.
7. Plot the graph respect to loss and iterations using .plot() method for Gradient Descent.


## Program:

```
/*
Program to implement the linear regression using gradient descent.
Developed by: Vidhiya Lakshmi S
RegisterNumber: 212223230238 
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
```
```
df=pd.read_csv('/content/student_scores.csv')
print(df.head())
print(df.tail())
```

## Output:

![image](https://github.com/user-attachments/assets/132ed7d6-78fc-4ca8-9b7b-abd3a803e6c5)

```
X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,1].values
print(Y)

```

## Output:
![image](https://github.com/user-attachments/assets/aac7cbe8-fcca-46ff-b24c-ffa36df669f2)

```
m=0
c=0
L=0.001
epochs=5000
n=float(len(X))
error=[]
for i in range(epochs):
  Y_pred=m*X+c
  D_m=(-2/n)*sum(X*(Y-Y_pred))
  D_c=(-2/n)*sum(Y-Y_pred)
  m=m-L*D_m
  c=c-L*D_c
  error.append(sum(Y-Y_pred)**2)
print(m,c)
```

## Output:
![image](https://github.com/user-attachments/assets/7d68bacd-addd-4d07-83e6-988bf5a8e040)

```
type(error)
print(len(error))
plt.plot(range(0,epochs),error)

```
## Output:

5000

![image](https://github.com/user-attachments/assets/852b0f87-ee71-4d53-84dc-b3a39fec3b58)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
