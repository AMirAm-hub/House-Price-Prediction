import numpy as np 
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense , Dropout
from tensorflow.keras.optimizers import Adam

iris_data = load_iris()
x = iris_data.data
y = iris_data.target
# print(y.shape)
y = keras.utils.to_categorical(y, num_classes=3)
# print(y.shape)
assert y.shape == (150,3)
train_x , test_x , train_y , test_y = train_test_split(x,y,test_size=0.20)

model = Sequential()
model.add(Dense(10,activation='relu',input_dim = 4))
model.add(Dense(10,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3,activation='softmax'))
# print(model.summary())

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

h=model.fit(train_x,train_y,
        validation_data=(test_x, test_y),
        batch_size=5,
        epochs=200,
          )

plt.plot(h.history['accuracy'], label='Training accuracy')
# plt.plot(h.history['loss'], label='Training loss')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

result = model.evaluate(test_x,train_y)