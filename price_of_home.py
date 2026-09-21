import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

data = pd.read_csv('/Users/amiram/Desktop/tensorflow-keras/AmesHousing.csv')
data_x = data[['Gr Liv Area','Bedroom AbvGr','Full Bath','Year Built',]]
data_y = data[['SalePrice']]
# print(data.head())
x_train , x_test , y_train , y_test = train_test_split(data_x,data_y,random_state=11)
# print(x_train.shape)
rgg = LinearRegression()
rgg.fit(x_train,y_train)
# print(rgg.score(x_test,y_test)) it's R^2 is --> 0.7365247664709236


living_space = int(input("enter your LivingـSpace (it is based on foot) :"))
bedrooms = int(input('enter your Number of Bedrooms :'))
bathrooms = int((input('enter your Number of Full Bathrooms :')))
year = int(input('enter your Year of Construction (it is based on year) :'))
new_house = pd.DataFrame({
    'Gr Liv Area': [living_space],
    'Bedroom AbvGr': [bedrooms],
    'Full Bath': [bathrooms],
    'Year Built': [year]
})
prediction = rgg.predict(new_house) 
print(f"{prediction.round().item()}$")

# plt.scatter(data['Gr Liv Area'], data['SalePrice'])

# plt.xlabel('Living Area (sq ft)')
# plt.ylabel('Sale Price ($)')
# plt.title('Living Area vs Sale Price')

# plt.show()