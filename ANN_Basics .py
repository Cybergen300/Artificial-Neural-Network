import os 
path = "/your path"
os.chdir(path)
import keras
from keras.datasets import mnist 
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np 
from keras.applications.vgg19 import VGG19
from keras.applications import InceptionV3
from keras.applications import MobileNet


#ResNet 50 model 
#===============

model = ResNet50(weights='imagenet')

img_path = 'scotch.png'
img = image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)
x= np.expand_dims(x, axis=0)
x=preprocess_input(x)

preds = model.predict(x)

#decode the results into a list of tuples (class, description, probability)
#(one such list for each sample in the batch)

#features = model.predict(x)

print('Predicted:', decode_predictions(preds, top=3)[0])


#Fine Tuning of the ResNet 50 model 
#==================================


batch_size = 50
num_classes = 10
epochs = 6

#input image dimensions
img_rows, img_cols = 28,28

#the data split between train and test sets 
(x_train, y_train), (x_test, y_test)= mnist.load_data()
x_train = x_train[:30000]
y_train = y_train[:30000]

#choose the right format depending on the used framework in our data
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)

else : 
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test= x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape =(img_rows, img_cols, 1)



x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#convert class vectors to binary class matrices 
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#model 
model = Sequential()

#We add our layers
model.add(Dense(32,
                 activation='relu',
                 input_shape=input_shape))

model.add(Dense(30, activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

#We train our model on our data with the help of the fit method
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=8,
          verbose=1,
validation_data=(x_test, y_test))

#We compute the accuracy of our model and show the results
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



