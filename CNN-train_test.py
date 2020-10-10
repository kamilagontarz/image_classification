import tensorflow
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np
import pickle
from tensorflow.keras import backend as K


pickle_in = open("X.pickle","rb")
train_images_1 = pickle.load(pickle_in)
train_images = np.array(train_images_1 )

pickle_in = open("y.pickle","rb")
train_labels_1 = pickle.load(pickle_in)
train_labels = np.array(train_labels_1)

pickle_in = open("X-test.pickle","rb")
test_images_1 = pickle.load(pickle_in)
test_images = np.array(test_images_1)

pickle_in = open("y-test.pickle","rb")
test_labels_1 = pickle.load(pickle_in)
test_labels = np.array(test_labels_1)

if K.image_data_format() == 'channels_first':
    train_images = train_images.reshape(train_images.shape[0], 3, 50, 50)
    test_images = test_images.reshape(test_images.shape[0], 3, 50, 50)
    input_shape = (3, 50, 50)
else:
    train_images = train_images.reshape(train_images.shape[0], 50, 50, 3)
    test_images = test_images.reshape(test_images.shape[0], 50, 50, 3)
    input_shape = (50, 50, 3)
    
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255

train_labels = tensorflow.keras.utils.to_categorical(train_labels, 5)
test_labels = tensorflow.keras.utils.to_categorical(test_labels, 5)

print("")
print("[STATUS] Przygotowanie modelu...")
print("")

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))


model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

#model.summary()


print("")
print("[STATUS] Kompilacja modelu...")
print("")

from tensorflow import keras

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(learning_rate=0.1),#'adam',
#              optimizer = 'rmsprop',
              metrics=['accuracy'],
              )


history = model.fit(train_images, train_labels,
                    batch_size=32,
                    #shuffle=True,
                    epochs=20,
                    verbose=2,
                    validation_data=(test_images, test_labels))

score = model.evaluate(test_images, test_labels, verbose=0)

print("")
print('____________________________')
print("")
print('Dane testowe - strata     : ', score[0])
print('Dane testowe - dokładność : ', score[1])
print("")