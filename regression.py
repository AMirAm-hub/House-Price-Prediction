import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import matplotlib.pyplot as plt

data_x = np.linspace(3,8,100)

rng = np.random.RandomState(6)
data_y =np.sin(4 * data_x) + data_x + rng.uniform(size=len(data_x))


# plt.plot(data_x,data_y,'o')
# plt.show()
# print(data_x.shape)
data_x = data_x[:,np.newaxis]   
# print(data_x.shape)
x_train,x_test,y_train,y_test = train_test_split(data_x,data_y)

regressor = LinearRegression()
regressor.fit(x_train,y_train)
# print("weight coefficientes : ",regressor.coef_)
# print("y-axis intercept : ",regressor.intercept_)
min_pt = data_x.min() * regressor.coef_[0] + regressor.intercept_
max_pt = data_x.max() * regressor.coef_[0] + regressor.intercept_

plt.plot([data_x.min(),data_x.max()],[min_pt,max_pt])
plt.plot(x_train,y_train,'o')
plt.show()