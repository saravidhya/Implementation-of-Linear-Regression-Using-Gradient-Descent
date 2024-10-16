# EXPERIMENT NO: 3
## Implementation-of-Linear-Regression-Using-Gradient-Descent
### NAME : AVINASH T
### REG NO: 212223230026
## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Import Libraries**: Load necessary libraries for data handling, metrics, and visualization.

2. **Load Data**: Read the dataset using `pd.read_csv()` and display basic information.

3. **Initialize Parameters**: Set initial values for slope (m), intercept (c), learning rate, and epochs.

4. **Gradient Descent**: Perform iterations to update `m` and `c` using gradient descent.

5. **Plot Error**: Visualize the error over iterations to monitor convergence of the model.
## Program & Output
```c
Program to implement the linear regression using gradient descent.
Developed by: AVINASH T
RegisterNumber: 212223230026

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def multivariate_linear_regression(X1, Y, learning_rate=0.01, num_iters=1000):
  X=np.c_[np.ones(len(X1)),X1]
  theta=np.zeros(X.shape[1]).reshape(-1,1)
  for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-Y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
 return theta
from sklearn.preprocessing import StandardScaler
df=pd.read_csv('/content/50_Startups.csv')
df.info()
```
![image](https://github.com/user-attachments/assets/70a22dbb-3ae2-4165-99ed-91c92d495884)
```c
df.head()
```
![image](https://github.com/user-attachments/assets/f33aecd8-ea50-4a51-a259-b828a4d8b0b1)
```c
df.tail()
```
![image](https://github.com/user-attachments/assets/b4328aad-f3ce-4849-af79-b9b962edcc64)
```c
x = (df.iloc[:, :-2].values)
y = (df.iloc[:, -1].values).reshape(-1,1)
print(x)
```
![image](https://github.com/user-attachments/assets/368bb892-ebce-419f-8014-17abbc1d7cda)
```c
print(y)
```
![image](https://github.com/user-attachments/assets/f0cf87b5-ae2a-4979-bea2-b1ac75c82190)
```c
scaler = StandardScaler()
x1=x.astype(float)
x1_scaled= scaler.fit_transform(x)
y1_scaled = scaler.fit_transform(y)
print(x1_scaled)
print(y1_scaled)
```
![image](https://github.com/user-attachments/assets/b6692819-46fd-4dcb-985e-b19609605a12)

![image](https://github.com/user-attachments/assets/ca969907-4e46-4acc-9dd7-8f098e02449e)
```c
theta=multivariate_linear_regression(x1_scaled,y1_scaled)
print(theta)
```
![image](https://github.com/user-attachments/assets/0a53c54e-25c5-4a5e-98b3-7ffa2b863f79)
```c
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value:{pre}")
```
![image](https://github.com/user-attachments/assets/eb003e10-9a64-493a-a2b6-48e24ce5955d)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
