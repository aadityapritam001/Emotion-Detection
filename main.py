import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
# import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

%matplotlib inline


import h5py
def load_dataset():
    train_dataset = h5py.File('datasets/train_happy.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_happy.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
#     print(classes)
#     print(train_set_y_orig.shape[0])
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

def creating_model(input_shape):
    X_input=Input(input_shape)
    X=ZeroPadding2D((3,3))(X_input)
    X=Conv2D(32,(5,5),strides=(1,1),name='conv1')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    X=MaxPooling2D((2,2),name='Maxpooling1')(X)
    X=Flatten()(X)
    X=Dense(1,activation='sigmoid',name='Fully_connected')(X)
    model=Model(inputs=X_input, outputs=X, name='emotion_detection_model')
    
    return model
    
    
happyModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
happyModel.fit(X_train, Y_train, epochs=6, batch_size=32)

pred=happyModel.predict(X_test[:3])
print("Loss:",pred[0])
print("Test Accuracy",pred[1])

print(happyModel.evaluate(X_test, Y_test, batch_size=32))

img_path='me.jpg'
img=image.load_img(img_path, target_size=(64,64))
imshow(img)

res=['Happy','Unhappy']

x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
x=preprocess_input(x)
result=happyModel.predict(x)
print(result)
for i in result:
    if(i[0]==1.):
        print('happy')
    else:
        print("unhappy")
