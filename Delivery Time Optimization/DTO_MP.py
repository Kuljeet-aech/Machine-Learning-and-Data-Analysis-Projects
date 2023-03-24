import numpy as np
from sklearn.model_selection import train_test_split
from DTO_analysis import data
x = np.array(data[["Delivery_person_Age",
                   "Delivery_person_Ratings",
                   "Distance"]])
y = np.array(data[["Time_taken(min)"]])

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.10,random_state=42)

from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(128,return_sequences=True, input_shape=(xtrain.shape[1],1)))
model.add(LSTM(64,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.summary

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(xtrain, ytrain, batch_size=1, epochs=9)

print("Food Delivery Time Prediction")
a = int(input("Age of Delivery Partner: "))
b = float(input("Ratings of Previous Deliveries: "))
c = int(input("Total Distance: "))

features = np.array([[a, b, c]])
print("Predicted Delivery Time in Minutes = ", model.predict(features))
