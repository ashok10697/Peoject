from __future__ import print_function
import keras
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import metrics
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt


from scipy.misc import imread, imresize, imshow

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)





batch_size = 100
num_classes = 10
epochs = 50

# input image dimensions
img_rows, img_cols = 64, 64
# The CIFAR10 images are RGB.
img_channels = 3



(x_train, y_train), (x_test, y_test) = np.load('load_data_32_32.npy')
x_train = x_train[:10000, : , : , :]
y_train = y_train[:10000]

#x_test = x_test[:1000,:,:,:]
#y_test = y_test[:1000]

x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)




model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=x_train.shape[1:],padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))



model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

model.save('my_model_1.h5')

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy', metrics.top_k_categorical_accuracy])



history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose = 2,
              validation_split = 0.10)


model.save('my_model_2.h5')




# list all data in history
print(history.history.keys())

# summarize history for accuracy
fig = plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig.savefig('accuracy_my_model_2.png')


# summarize history for loss
fig = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig.savefig('Loss_my_model_2.png')
