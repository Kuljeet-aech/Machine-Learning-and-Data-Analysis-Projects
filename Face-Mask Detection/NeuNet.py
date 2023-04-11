import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = np.load('DataSets//Face-Mask//data.npy')
target = np.load('DataSets//Face-Mask//target.npy')

nn = Sequential()
nn.add(Conv2D(filters=100, kernel_size=(2,2), activation='relu', input_shape = data.shape[1:]))
nn.add(MaxPooling2D(pool_size=(2,2)))
nn.add(Conv2D(filters=100, kernel_size=(2,2), activation='relu'))
nn.add(MaxPooling2D(pool_size=(2,2)))
nn.add(Flatten())
nn.add(Dropout(0.5))
nn.add(Dense(70, activation='relu'))
nn.add(Dense(40, activation='relu'))
nn.add(Dense(2, activation='softmax'))

nn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.2)

history = nn.fit(train_data, train_target, epochs= 25, validation_split= 0.2)

plt.plot(history.history['loss'], 'r', label = 'Training loss')
plt.plot(history.history['val_loss'], label = 'Validation loss')
plt.plot(history.history['acc'], 'b', label = 'Accuracy')
plt.plot(history.history['val_acc'], 'r', label = 'Validation Accuracy')
plt.xlabel('No. of epochs')
plt.ylabel('Loss')
plt.show()

print(nn.evaluate(test_data, test_target))
