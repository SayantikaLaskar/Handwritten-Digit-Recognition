# Natural-Language-Processing
Minor Project

import tensorflow as tf
(x_train, y_train), (x_test, y_test)= tf.keras.datasets.mnist.load_data() x_train= x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test= x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape= (28, 28, 1) x_train=x_train.astype('float32') x_test=x_test.astype('float32') x_train /=255
x_test /=255
import matp1otlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Deuse, Conv2D, Dropout, Flatten, MaxPooling2D model=Sequential()
model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape)) model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation=tf.nn.softmax))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) model.fit(x=x_train,y=y_train, epochs=10)
model.evaluate(x_test, y_test)
1mage_1ndex = 2853
pit  . 1mshow(x_test[1nage_1ndex] . reshape(28,  28) , cmap= ’ Greys ' )
predict = x_test[image_index].reshape(28,28)
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1)) print(pred.argmax())
